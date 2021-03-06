import os
import logging
import argparse
from logging import Logger
import napari

import h5py
import tifffile
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


def _predict(logger, config, model, grid, config_data, path_file):
    # Load dataset:
    image_raw = load_dataset(logger, config, config_data, path_file)

    # Check dataset size:
    config_patch_size = config_data['slice']['patch_size']
    patch_size = [min(i, j) for i, j in zip(config_patch_size, image_raw.shape)]
    if patch_size != config_patch_size:
        logger.info(f"Patch size {config_patch_size} specified in config is too big for dataset {path_file}")

    # Predict by blocks:
    halo = config_data['slice']['halo_size']
    blocking = nt.blocking([0, 0, 0], image_raw.shape, patch_size)
    number_of_instances, block_with_halo_sequence, probability_map_sequence = block_and_predict(model, image_raw, halo, blocking)

    # Check datatype capacity:
    if number_of_instances > np.iinfo(np.uint32).max:
        # this is put here to avoid checking at every iteration, may waste time if dataset is huge
        raise OverflowError(f"Number of pre-merged instances, {number_of_instances}, is greater than the maximum of np.uint32.")

    image_lab, image_prob = tile_blocks(grid, image_raw.shape, halo, blocking, block_with_halo_sequence, probability_map_sequence)

    # Save unstitched tiled blocks:
    Path(config_data['output_dir']).mkdir(parents=True, exist_ok=True)
    save_unstitched_tiles(config_data['format'], config_data['output_dir'], path_file, image_lab)

    # Stitching
    n_threads = config['stitching']['n_threads']
    overlap_threshold = config['stitching']['overlap_threshold']
    blocking = nt.blocking([0, 0, 0], image_lab.shape, patch_size)
    segmentation = stitch(halo, blocking, block_with_halo_sequence, image_lab, n_threads, overlap_threshold)

    # Save stitched blocks:
    output_dtype = np.dtype(config_data['output_dtype'])
    save_stitched_image(config_data['format'], config_data['output_dir'], path_file, image_prob, segmentation, output_dtype)


def save_stitched_image(dset_format, output_dir, path_file, image_prob, segmentation, output_dtype):
    if dset_format == 'hdf5':
        path_out_file = output_dir + os.path.splitext(os.path.basename(path_file))[0] + '_merged.h5'
        with h5py.File(path_out_file, 'w') as f:
            f.create_dataset(name="segmentation", data=segmentation.astype(output_dtype), compression='gzip')
            f.create_dataset(name="probability",  data=image_prob,                        compression='gzip')
    elif dset_format == 'tiff':
        path_out_file = output_dir + os.path.splitext(os.path.basename(path_file))[0] + '_merged.tif'
        tifffile.imwrite(path_out_file, data=segmentation.astype(output_dtype), imagej=True)
        path_out_file = output_dir + os.path.splitext(os.path.basename(path_file))[0] + '_prob.tif'
        tifffile.imwrite(path_out_file, data=image_prob,                        imagej=True)


def stitch(halo, blocking, block_with_halo_sequence, image_lab, n_threads, overlap_threshold):
    overlap_dimensions, overlap_dict = compute_overlaps(block_with_halo_sequence, blocking, halo, n_threads)
    node_labels, block_offsets = stitch_segmentations_by_overlap(block_with_halo_sequence,
                                                                 overlap_dimensions,
                                                                 overlap_dict,
                                                                 overlap_threshold,
                                                                 n_threads)
    segmentation = merge_segmentation(image_lab, node_labels, block_offsets, blocking, n_threads)
    segmentation, _, _ = relabel_sequential(segmentation)  # 30x faster, len(forward_map) != len(inverse_map)
    return segmentation


def save_unstitched_tiles(dset_format, output_dir, path_file, image_lab):
    if dset_format == 'hdf5' and config.get('save_tiled'):
        path_out_file = output_dir + os.path.splitext(os.path.basename(path_file))[0] + '_tiled.h5'
        with h5py.File(path_out_file, 'w') as f:
            f.create_dataset(name='unstitched', data=image_lab.astype("uint16"), compression='gzip')


def tile_blocks(grid, image_shape, halo, blocking, block_with_halo_sequence, probability_map_sequence):
    image_lab = np.zeros(image_shape, dtype=np.uint32)
    image_prob = np.zeros(image_shape, dtype=np.float32)

    for block_id in trange(blocking.numberOfBlocks):
        this_block = blocking.getBlockWithHalo(block_id, halo)
        this_slice = tuple(slice(beg, end) for beg, end in zip(this_block.innerBlock.begin,
                                                               this_block.innerBlock.end))
        this_slice_local = tuple(slice(beg, end) for beg, end in zip(this_block.innerBlockLocal.begin,
                                                                     this_block.innerBlockLocal.end))
        image_lab[this_slice] = block_with_halo_sequence[block_id][this_slice_local]
        image_prob[this_slice] = zoom(probability_map_sequence[block_id], grid)[this_slice_local]
    return image_lab, image_prob


def block_and_predict(model, image_raw, halo, blocking):
    number_of_instances = 0
    block_with_halo_sequence = []
    probability_map_sequence = []
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

        probmap, _ = model.predict(img)
        probability_map_sequence.append(probmap)
    return number_of_instances, block_with_halo_sequence, probability_map_sequence


def load_dataset(logger, config, config_data, path_file):
    dset_format = config_data['format']
    if utils.is_hdf5(dset_format):
        # Load HDF5 dataset:
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
            else:
                logger.warn("No voxel information provided.")
    elif utils.is_tiff(dset_format):
        # Load TIFF dataset:
        image_raw = tifffile.imread(path_file).squeeze()
        image_raw = normalize(image_raw, 1, 99.8, axis=config['normalisation']['axis_norm'])
    return image_raw


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
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
    grid = config['stardist']['grid']

    # List Datasets:
    logger.info("Scanning datasets, please make sure file extensions are in lower case.")
    config_data = config['data']
    paths_dataset = config_data['path']
    if not isinstance(paths_dataset, list):
        raise TypeError("config['data']['path'] has to be a list of file/dir path(s)!")
    paths_file_dataset = utils.get_dataset_file_paths(paths_dataset, config_data['format'])
    logger.info(f"Found the following files: \n\t{paths_file_dataset}")

    # Prediction:
    for dataset_idx, path_file in enumerate(paths_file_dataset):
        logger.info(f"Predicting dataset {dataset_idx}, {len(paths_file_dataset)} datasets in total:")

        _predict(logger, config, model, grid, config_data, path_file)
