import json
import os
import time
from concurrent import futures

import h5py
import numpy as np
import nifty.ufd as nufd
import nifty.ground_truth as ngt
import nifty.tools as nt
import vigra

import elf.segmentation.features as feats
from elf.segmentation.multicut import get_multicut_solver, compute_edge_costs
from elf.segmentation.utils import normalize_input
from tqdm import tqdm
from natsort import natsorted


#
# general purpose functionality
#


def intersect_bounding_boxes(bb_a, bb_b):
    def _intersect_dim(b_a, b_b):
        b0, b1 = (b_a, b_b) if b_a.start < b_b.start else (b_b, b_a)
        start = b1.start
        if b0.stop < start:
            return None
        elif b0.stop < b1.stop:
            stop = b0.stop
        elif b0.stop >= b1.stop:
            stop = b1.stop
        return start, stop

    intersections = [_intersect_dim(b_a, b_b) for b_a, b_b in zip(bb_a, bb_b)]
    if any(intersection is None for intersection in intersections):
        return None
    bb = tuple(slice(start, stop) for start, stop in intersections)
    return bb


def segment_blocks(boundary_map, ws, mc_solver, blocking, halo, n_threads):
    segmentation = np.zeros_like(ws)

    def _segment_block(block_id):
        block = blocking.getBlockWithHalo(block_id, halo)
        bb = tuple(
            slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end)
        )
        bd = normalize_input(boundary_map[bb])
        seg = ws[bb]
        assert bd.shape == seg.shape
        seg, max_id, _ = vigra.analysis.relabelConsecutive(seg, start_label=0, keep_zeros=False)

        rag = feats.compute_rag(seg, int(max_id + 1), n_threads=1)
        edge_features = feats.compute_boundary_mean_and_length(rag, bd, n_threads=1)[:, 0]
        costs = compute_edge_costs(edge_features)

        node_labels = mc_solver(rag, costs)
        seg = feats.project_node_labels_to_pixels(rag, node_labels, n_threads=1)

        bb = tuple(
            slice(beg, end) for beg, end in zip(block.innerBlock.begin, block.innerBlock.end)
        )
        bb_local = tuple(
            slice(beg, end) for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end)
        )
        segmentation[bb] = seg[bb_local]
        return seg

    with futures.ThreadPoolExecutor(n_threads) as tp:
        block_segmentation = list(tqdm(
            tp.map(_segment_block, range(blocking.numberOfBlocks)),
            total=blocking.numberOfBlocks,
            desc="Segment blocks"
        ))

    return segmentation, block_segmentation


def segment_blocks_cached(boundary_map, ws, mc_solver, blocking, halo, n_threads, tmp_folder):
    os.makedirs(tmp_folder, exist_ok=True)

    seg_path = os.path.join(tmp_folder, 'seg.h5')
    if os.path.exists(seg_path):
        with h5py.File(seg_path, 'r') as f:
            seg = f['data'][:]

        def _load_block(block_id):
            block_path = os.path.join(tmp_folder, f'block_{block_id}.h5')
            assert os.path.exists(block_path)
            with h5py.File(block_path, 'r') as f:
                return f['data'][:]

        with futures.ThreadPoolExecutor(n_threads) as tp:
            block_segmentation = list(tqdm(
                tp.map(_load_block, range(blocking.numberOfBlocks)),
                total=blocking.numberOfBlocks,
                desc="Load blocks"
            ))
        return seg, block_segmentation

    t_seg = time.time()
    segmentation, block_segmentation = segment_blocks(boundary_map, ws, mc_solver, blocking, halo, n_threads)
    t_seg = time.time() - t_seg
    with h5py.File(seg_path, 'w') as f:
        f.create_dataset('data', data=segmentation)

    def _write_block(block_id):
        block_path = os.path.join(tmp_folder, f'block_{block_id}.h5')
        with h5py.File(block_path, 'w') as f:
            f.create_dataset('data', data=block_segmentation[block_id])

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_write_block, range(blocking.numberOfBlocks)),
            total=blocking.numberOfBlocks,
            desc="Write blocks"
        ))

    with open(os.path.join(tmp_folder, 't_seg.json'), 'w') as f:
        json.dump({'t-seg': t_seg}, f)

    return segmentation, block_segmentation


def get_block_offsets(blocks):
    offsets = [block.max() + 1 for block in blocks]
    last_max_id = offsets[-1]
    offsets = np.roll(offsets, 1)
    offsets = np.cumsum(offsets).astype('uint64')
    number_of_nodes = offsets[-1] + last_max_id
    return number_of_nodes, offsets


def apply_offsets(segmentation, blocking, offsets, n_threads):
    assert len(offsets) == blocking.numberOfBlocks

    def _apply_block(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        segmentation[bb] = (segmentation[bb] + offsets[block_id])

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tp.map(_apply_block, range(blocking.numberOfBlocks))

    return segmentation


#
# functionalty for multicut stitching
#

def solve_overlap_problems(boundary_map, segmentation, blocking,
                           mc_solver, halo, n_threads):
    def _solve_problem(bb):
        bd = normalize_input(boundary_map[bb])
        seg = segmentation[bb]
        assert bd.shape == seg.shape

        seg, max_id, mapping = vigra.analysis.relabelConsecutive(seg, start_label=0, keep_zeros=False)

        rag = feats.compute_rag(seg, int(max_id + 1), n_threads=1)
        edge_features = feats.compute_boundary_mean_and_length(rag, bd, n_threads=1)[:, 0]
        costs = compute_edge_costs(edge_features)
        node_labels = mc_solver(rag, costs)

        uv_ids = rag.uvIds()
        merged_edges = node_labels[uv_ids[:, 0]] == node_labels[uv_ids[:, 1]]

        mapping = {v: k for k, v in mapping.items()}
        uv_ids = nt.takeDict(mapping, uv_ids)

        merge_pairs = uv_ids[merged_edges]
        return merge_pairs

    def _solve_problems(block_id):
        this_block = blocking.getBlockWithHalo(block_id, halo)
        this_bb = tuple(
            slice(beg, end) for beg, end in zip(this_block.outerBlock.begin, this_block.outerBlock.end)
        )

        merge_pairs = []
        for axis in range(3):
            ngb_id = blocking.getNeighborId(block_id, axis, True)
            if ngb_id == -1:
                continue

            other_block = blocking.getBlockWithHalo(ngb_id, halo)
            other_bb = tuple(
                slice(beg, end) for beg, end in zip(other_block.outerBlock.begin, other_block.outerBlock.end)
            )
            global_bb = intersect_bounding_boxes(this_bb, other_bb)
            assert global_bb is not None

            this_merge_pairs = _solve_problem(global_bb)
            if this_merge_pairs.size:
                merge_pairs.append(this_merge_pairs)

        if merge_pairs:
            merge_pairs = np.concatenate(merge_pairs)
        else:
            merge_pairs = None
        return merge_pairs

    # with futures.ThreadPoolExecutor(n_threads) as tp:
    with futures.ThreadPoolExecutor(1) as tp:
        merge_pairs = list(tqdm(
            tp.map(_solve_problems, range(blocking.numberOfBlocks)),
            total=blocking.numberOfBlocks,
            desc="Solve overlap problems"
        ))

    merge_pairs = np.concatenate([mp for mp in merge_pairs if mp is not None])
    merge_pairs = np.unique(merge_pairs, axis=0)
    return merge_pairs


def multicut_stitching(boundary_map, ws,
                       solver, solver_kwargs,
                       tmp_folder, block_shape, halo, n_threads):
    assert boundary_map.shape == ws.shape
    blocking = nt.blocking([0, 0, 0], ws.shape, block_shape)

    # segment individual blocks with halo
    mc_solver = get_multicut_solver(solver, **solver_kwargs)
    segmentation, block_segmentation = segment_blocks_cached(boundary_map, ws, mc_solver,
                                                             blocking, halo, n_threads, tmp_folder)

    # make sure we have unique seg ids per block
    n_nodes, offsets = get_block_offsets(block_segmentation)
    segmentation = apply_offsets(segmentation, blocking, offsets, n_threads)

    t_merge = time.time()
    merge_pairs = solve_overlap_problems(boundary_map, segmentation, blocking,
                                         mc_solver, halo, n_threads)
    t_merge = time.time() - t_merge
    with open(os.path.join(tmp_folder, 't_merge.json'), 'w') as f:
        json.dump({'t-merge': t_merge}, f)

    ufd = nufd.ufd(n_nodes)
    ufd.merge(merge_pairs)
    node_labels = ufd.elementLabeling()
    segmentation = nt.take(node_labels, segmentation)
    return segmentation


#
# functionality for greedy stitching
#

def greedy_stitching(boundary_map, ws,
                     solver, solver_kwargs,
                     tmp_folder, block_shape, halo,
                     threshold, n_threads):
    assert boundary_map.shape == ws.shape
    blocking = nt.blocking([0, 0, 0], ws.shape, block_shape)

    # segment individual blocks with halo
    mc_solver = get_multicut_solver(solver, **solver_kwargs)
    segmentation, block_segmentation = segment_blocks_cached(boundary_map, ws, mc_solver,
                                                             blocking, halo, n_threads, tmp_folder)

    # make sure we have unique seg ids per block
    n_nodes, offsets = get_block_offsets(block_segmentation)
    segmentation = apply_offsets(segmentation, blocking, offsets, n_threads)

    rag_save_path = os.path.join(tmp_folder, 'rag.h5')
    if os.path.exists(rag_save_path):
        with h5py.File(rag_save_path, 'r') as f:
            uv_ids = f['edges'][:]
            stitch_edges = f['stitch_edges'][:]
            features = f['features'][:]

    else:
        t_graph = time.time()
        rag = feats.compute_rag(segmentation, n_threads=n_threads)
        stitch_edges = feats.get_stitch_edges(rag, segmentation, block_shape, n_threads=n_threads,
                                              verbose=True)
        features = feats.compute_boundary_mean_and_length(rag, normalize_input(boundary_map), n_threads=n_threads)[:, 0]
        t_graph = time.time() - t_graph

        with open(os.path.join(tmp_folder, 't_rag.json'), 'w') as f:
            json.dump({'t-rag': t_graph}, f)

        uv_ids = rag.uvIds()
        with h5py.File(rag_save_path, 'w') as f:
            f.create_dataset('edges', data=uv_ids)
            f.create_dataset('stitch_edges', data=stitch_edges)
            f.create_dataset('features', data=features)

    ufd = nufd.ufd(n_nodes)
    merge_edges = np.logical_and(features < threshold, stitch_edges)
    assert len(merge_edges) == len(uv_ids)
    ufd.merge(uv_ids[merge_edges])
    node_labels = ufd.elementLabeling()
    segmentation = nt.take(node_labels, segmentation)
    return segmentation


#
# functionality for overlap based stitching,
# based on: https://github.com/constantinpape/cremi_tools/blob/master/cremi_tools/stitching/stitch_by_overlap.py
#

def merge_blocks(overlap_ids, overlaps,
                 overlap_dimensions,
                 offsets, ovlp_threshold,
                 ignore_background=True):
    id_a, id_b = overlap_ids
    ovlp_a, ovlp_b = overlaps
    offset_a, offset_b = offsets[id_a], offsets[id_b]
    assert ovlp_a.shape == ovlp_b.shape, "%s, %s" % (str(ovlp_a.shape), str(ovlp_b.shape))

    ovlp_dim = overlap_dimensions[(id_a, id_b)]
    # find the ids ON the actual block boundary
    ovlp_len = ovlp_a.shape[ovlp_dim]
    ovlp_dim_begin = ovlp_len // 2 if ovlp_len % 2 == 1 else ovlp_len // 2 - 1
    ovlp_dim_end = ovlp_len // 2 + 1
    boundary = tuple(slice(None) if i != ovlp_dim else
                     slice(ovlp_dim_begin, ovlp_dim_end) for i in range(3))

    # measure all overlaps
    overlaps_ab = ngt.overlap(ovlp_a, ovlp_b)
    overlaps_ba = ngt.overlap(ovlp_b, ovlp_a)
    node_assignment = []

    # find the ids ON the actual block boundary
    segments_a = np.unique(ovlp_a[boundary])
    # if ignore_background and segments_a[0] == 0 and len(segments_a) > 0:
    #     segments_a = segments_a[1:]
    segments_b = np.unique(ovlp_b[boundary])
    # if ignore_background and segments_b[0] == 0 and len(segments_b) > 0:
    #     segments_b = segments_b[1:]

    for seg_a in segments_a:

        ovlp_seg_a, counts_seg_a = overlaps_ab.overlapArraysNormalized(seg_a, sorted=True)
        seg_b = ovlp_seg_a[0]

        ovlp_seg_b, counts_seg_b = overlaps_ba.overlapArraysNormalized(seg_b, sorted=True)
        if ovlp_seg_b[0] != seg_a or seg_b not in segments_b:
            continue

        ovlp_measure = (counts_seg_a[0] + counts_seg_b[0]) / 2.
        if ovlp_measure > ovlp_threshold:
            node_assignment.append([seg_a + offset_a, seg_b + offset_b])

    if node_assignment:
        return np.array(node_assignment, dtype='uint32')
    else:
        return None


def stitch_segmentations_by_overlap(blocks,
                                    overlap_dimensions,
                                    overlap_dict,
                                    ovlp_threshold,
                                    n_threads):
    # validate all inputs
    assert isinstance(blocks, list)
    assert all(isinstance(block, np.ndarray) for block in blocks)
    assert isinstance(overlap_dimensions, dict)
    assert isinstance(overlap_dict, dict)

    # (this could be parallelised, but rt doesn't matter compared to solving the mc sub-problems)
    # find the offsets for each block and the max node id
    number_of_nodes, offsets = get_block_offsets(blocks)

    # (this could be parallelised, but rt doesn't matter compared to solving the mc sub-problems)
    # iterate over the overlaps to find the node assignments
    node_assignment = []
    for ovlp_ids, overlaps in overlap_dict.items():
        node_assignment.append(merge_blocks(ovlp_ids, overlaps,
                                            overlap_dimensions,
                                            offsets, ovlp_threshold))
    node_assignment = np.concatenate([na for na in node_assignment if na is not None],
                                     axis=0)

    # merge nodes with union find
    ufd = nufd.ufd(number_of_nodes)
    ufd.merge(node_assignment)
    node_labels = ufd.elementLabeling()
    vigra.analysis.relabelConsecutive(node_labels, out=node_labels, start_label=1, keep_zeros=True)

    return node_labels, offsets


def compute_overlaps(block_segmentation, blocking, halo, n_threads):
    overlap_dimensions = {}
    overlap_dict = {}

    for block_id in range(blocking.numberOfBlocks):
        this_block = blocking.getBlockWithHalo(block_id, halo)
        this_bb = tuple(
            slice(beg, end) for beg, end in zip(this_block.outerBlock.begin, this_block.outerBlock.end)
        )
        for axis in range(3):
            ngb_id = blocking.getNeighborId(block_id, axis, True)
            if ngb_id == -1:
                continue

            overlap_dimensions[(block_id, ngb_id)] = axis

            other_block = blocking.getBlockWithHalo(ngb_id, halo)
            other_bb = tuple(
                slice(beg, end) for beg, end in zip(other_block.outerBlock.begin, other_block.outerBlock.end)
            )
            global_overlap = intersect_bounding_boxes(this_bb, other_bb)
            assert global_overlap is not None
            this_local_bb = tuple(
                slice(bb.start - off, bb.stop - off) for bb, off in zip(global_overlap, this_block.outerBlock.begin)
            )
            other_local_bb = tuple(
                slice(bb.start - off, bb.stop - off) for bb, off in zip(global_overlap, other_block.outerBlock.begin)
            )
            # add the overlaps of the two blocks as tuple to the dict
            overlap_dict[(block_id, ngb_id)] = (
                block_segmentation[block_id][this_local_bb],
                block_segmentation[ngb_id][other_local_bb],
            )

    return overlap_dimensions, overlap_dict


def merge_segmentation(segmentation, node_labels, block_offsets, blocking, n_threads):
    assert len(block_offsets) == blocking.numberOfBlocks

    def _merge_block(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        seg = segmentation[bb]
        seg += block_offsets[block_id]
        segmentation[bb] = nt.take(node_labels, seg)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_merge_block, range(blocking.numberOfBlocks)),
            total=blocking.numberOfBlocks,
            desc="Merge segmentation"
        ))

    return segmentation


def overlap_stitching(boundary_map, ws,
                      solver, solver_kwargs,
                      tmp_folder, block_shape, halo,
                      overlap_threshold, n_threads):
    assert boundary_map.shape == ws.shape
    blocking = nt.blocking([0, 0, 0], ws.shape, block_shape)

    # segment individual blocks with halo
    mc_solver = get_multicut_solver(solver, **solver_kwargs)
    segmentation, block_segmentation = segment_blocks_cached(boundary_map, ws, mc_solver,
                                                             blocking, halo, n_threads, tmp_folder)

    overlap_dimensions, overlap_dict = compute_overlaps(block_segmentation, blocking, halo, n_threads)
    node_labels, block_offsets = stitch_segmentations_by_overlap(block_segmentation,
                                                                 overlap_dimensions,
                                                                 overlap_dict,
                                                                 overlap_threshold,
                                                                 n_threads)

    segmentation = merge_segmentation(segmentation, node_labels, block_offsets, blocking, n_threads)
    return segmentation


if __name__ == "__main__":
    halo = [16, 48, 48]
    n_threads = 32
    overlap_threshold = 0.5

    block_segmentation = []
    with h5py.File("/g/kreshuk/yu/Outputs/Ovules2021/omp_threads/54_list.h5", 'r') as f:
        for i in natsorted(f.keys()):
            block_segmentation.append(f[i][:])
    with h5py.File("/g/kreshuk/yu/Outputs/Ovules2021/omp_threads/54_list_cat.h5", 'r') as f:
        segmentation = f['segmentation'][:]
    print(len(block_segmentation), segmentation.shape)

    blocking = nt.blocking([0, 0, 0], segmentation.shape, [32, 96, 96])

    overlap_dimensions, overlap_dict = compute_overlaps(block_segmentation, blocking, halo, n_threads)
    node_labels, block_offsets = stitch_segmentations_by_overlap(block_segmentation,
                                                                 overlap_dimensions,
                                                                 overlap_dict,
                                                                 overlap_threshold,
                                                                 n_threads)
    segmentation = merge_segmentation(segmentation, node_labels, block_offsets, blocking, n_threads)

    for i_new, i_old in enumerate(tqdm(np.unique(segmentation))):
        segmentation[segmentation == i_old] = i_new
    # segmentation, _, _ =

    with h5py.File("/g/kreshuk/yu/Outputs/Ovules2021/omp_threads/54_merge_16bit.h5", 'w') as f:
        # f.create_dataset("segmentation", data=segmentation, compression='gzip')
        f.create_dataset("segmentation", data=segmentation.astype(np.uint16), compression='gzip')

    print(np.max(segmentation))
