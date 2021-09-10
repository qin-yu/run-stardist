import os
import logging
import argparse
from logging import Logger

import yaml
import h5py
import numpy as np
import nifty.tools as nt
from pathlib import Path
from stardist.models import StarDist2D, StarDist3D
from skimage.segmentation import relabel_sequential
from scipy.ndimage import zoom
from csbdeep.utils import normalize

from tqdm import tqdm, trange

import utils
from stitch import compute_overlaps, stitch_segmentations_by_overlap, merge_segmentation


def load_model(config_stardist):
    if config_stardist['model_type'] == 'StarDist3D':
        Model = StarDist3D
    elif config_stardist['model_type'] == 'StarDist2D':
        Model = StarDist2D
    else:
        raise ValueError(f"'{config_stardist['model_type']}' is not a valid StarDist model type!")
    return Model(None, name=config_stardist['model_name'], basedir=config_stardist['model_dir'])


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.info("Running predict.py")
    utils.get_gpu_info()

    # Parse Arguments:
    args = utils.parse_arguments()

    # Load Config:
    config = utils.load_config(args.config_file)

    # TensorFlow Settings:
    utils.set_tf_op_parallelism_threads(config)

    # Load Model:
    model = load_model(config['stardist'])

    # List Datasets:
    logger.info("Scanning datasets")
    config_data = config['data']
    paths_dataset = config_data['path']
    if not isinstance(paths_dataset, list):
        raise TypeError("config['data']['path'] has to be a list of file/dir path(s)!")
    paths_file_dataset = utils.get_dataset_file_paths(paths_dataset)
    logger.info(f"Found the following files: \n\t{paths_file_dataset}")

    # Prediction:
    for dataset_idx, path_file in enumerate(paths_file_dataset):
        logger.info(f"Predicting dataset {dataset_idx}, {len(paths_file_dataset)} datasets in total:")
        with h5py.File(path_file, 'r') as f:
            dset = f[config_data['name']]
            if len(dset.shape) not in [2, 3]:
                raise ValueError(f"Image has shape {dset.shape}, expecting a 2D or 3D image!")
            image_raw = dset[:].squeeze()
            image_raw = normalize(image_raw, 1, 99.8, axis=config['normalisation']['axis_norm'])
            if 'voxel_size_um' in dset.attrs and config.get('rescale'):
                voxel_size_train = np.array(config['rescale']['voxel_size'])
                voxel_size = np.array(dset.attrs['voxel_size_um'])
                image_raw_dtype = image_raw.dtype
                rescale_ratio = voxel_size / voxel_size_train
                image_raw = zoom(image_raw, rescale_ratio)
                assert image_raw_dtype == image_raw.dtype, "Bug in rescaling"

        config_patch_size = config_data['slice']['patch_size']
        patch_size = [min(i, j) for i, j in zip(config_patch_size, image_raw.shape)]
        if patch_size != config_patch_size:
            logger.info(f"Patch size {config_patch_size} specified in config is too big for dataset {path_file}")

        halo = config_data['slice']['halo_size']

        number_of_instances = 0
        block_with_halo_sequence = []
        blocking = nt.blocking([0, 0, 0], image_raw.shape, patch_size)
        for block_id in trange(blocking.numberOfBlocks):
            this_block = blocking.getBlockWithHalo(block_id, halo)
            this_slice = tuple(slice(beg, end) for beg, end in zip(this_block.outerBlock.begin,
                                                                   this_block.outerBlock.end))
            img = image_raw[this_slice]
            labels, details = model.predict_instances(img)
            labels = labels.astype(np.uint32)
            labels[labels > 0] = labels[labels > 0] + number_of_instances
            number_of_instances += len(details['dist'])  # exclude background i.e. len(np.unique(labels)) - 1
            block_with_halo_sequence.append(labels)  # StarDist prediction gives various dtype, e.g. uint16/int32
        if number_of_instances > np.iinfo(np.uint32).max:
            # this is put here to avoid checking at every iteration, may waste time if dataset is huge
            raise OverflowError(
                f"Number of pre-merged instances {number_of_instances} is greater than the maximum of np.uint32.")

        image_lab = np.zeros(image_raw.shape, dtype=np.uint32)
        for block_id in trange(blocking.numberOfBlocks):
            this_block = blocking.getBlockWithHalo(block_id, halo)
            this_slice = tuple(slice(beg, end) for beg, end in zip(this_block.innerBlock.begin,
                                                                   this_block.innerBlock.end))
            this_slice_local = tuple(slice(beg, end) for beg, end in zip(this_block.innerBlockLocal.begin,
                                                                         this_block.innerBlockLocal.end))
            image_lab[this_slice] = block_with_halo_sequence[block_id][this_slice_local]

        # Stitching
        n_threads = config['stitching']['n_threads']
        overlap_threshold = config['stitching']['overlap_threshold']
        blocking = nt.blocking([0, 0, 0], image_lab.shape, patch_size)
        overlap_dimensions, overlap_dict = compute_overlaps(block_with_halo_sequence, blocking, halo, n_threads)
        node_labels, block_offsets = stitch_segmentations_by_overlap(block_with_halo_sequence,
                                                                     overlap_dimensions,
                                                                     overlap_dict,
                                                                     overlap_threshold,
                                                                     n_threads)
        segmentation = merge_segmentation(image_lab, node_labels, block_offsets, blocking, n_threads)
        segmentation, _, _ = relabel_sequential(segmentation)  # 30x faster, len(forward_map) != len(inverse_map)

        number_of_unique_labels = segmentation.max() + 1  # this should include bg thus +1 here, i.e. len(inverse_map)
        if number_of_unique_labels <= np.iinfo(np.dtype(config_data['output_dtype'])).max + 1:
            logger.info(
                f"User-specified type {config_data['output_dtype']} is used: {number_of_unique_labels} unique labels")
            output_dtype = np.dtype(config_data['output_dtype'])
        else:
            assert number_of_unique_labels <= np.iinfo(np.dtype(np.uint32)).max + 1, \
                "This shouldn't happen because labels are uint32 before merging and relabeling"
            logger.warning(
                f"User-specified type {config_data['output_dtype']} max exceeded, using uint32: {number_of_unique_labels} unique labels")
            output_dtype = np.uint32

        path_out_file = config_data['output_dir'] + os.path.splitext(os.path.basename(path_file))[0] + '_merged.h5'
        Path(config_data['output_dir']).mkdir(parents=True, exist_ok=True)
        with h5py.File(path_out_file, 'w') as f:
            f.create_dataset(name="segmentation",
                             data=segmentation.astype(output_dtype),
                             compression='gzip')
