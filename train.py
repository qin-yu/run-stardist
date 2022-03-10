import logging
import os
import h5py
import numpy as np
from glob import glob
from natsort import natsorted
from stardist import calculate_extents, gputools_available, Rays_GoldenSpiral
from stardist.models import Config3D, StarDist3D
from csbdeep.utils import normalize

import utils


def _gen_indices(i, k, s):
    assert i >= k, 'Sample size has to be bigger than the patch size'
    for j in range(0, i - k + 1, s):
        yield j
    if j + k < i:
        yield i - k


def get_slices_list(image_shape, patch_shape, stride_sizes):
    i_z, i_y, i_x = image_shape
    k_z, k_y, k_x = patch_shape
    s_z, s_y, s_x = stride_sizes

    assert i_z >= k_z and i_y >= k_y and i_x >= k_x

    slices = []
    z_steps = _gen_indices(i_z, k_z, s_z)
    for z in z_steps:
        y_steps = _gen_indices(i_y, k_y, s_y)
        for y in y_steps:
            x_steps = _gen_indices(i_x, k_x, s_x)
            for x in x_steps:
                slice_idx = (slice(z, z + k_z), slice(y, y + k_y), slice(x, x + k_x))
                slices.append(slice_idx)

    return slices


def random_fliprot(img, mask, axis=None):
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)

    assert img.ndim >= mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(transpose_axis)
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1, 2))
    x = random_intensity_change(x)
    return x, y


def load_datasets(logger, config, paths_file_dataset, patch_size, stride_sizes):
    patches_raw_all = []
    patches_lab_all = []
    extents_all = []
    for dataset_idx, path_file_dataset in enumerate(paths_file_dataset):
        logger.info(f"Loading dataset: {path_file_dataset}")
        with h5py.File(path_file_dataset, 'r') as f:
            try:
                image_raw = f['raw'][:]
                image_raw = normalize(image_raw, 1, 99.8, axis=config['axis_norm'])
            except KeyError:
                raise KeyError(f"The HDF5 file {path_file_dataset} doesn't contain a dataset named 'raw'") from None
            try:
                image_lab = f['label']
            except KeyError:
                raise KeyError(f"The HDF5 file {path_file_dataset} doesn't contain a dataset named 'label'") from None
            if image_raw.shape != image_lab.shape:
                raise IndexError(f"'raw' has shape {image_raw.shape} but 'label' has shape {image_lab.shape}")

            extents = calculate_extents(image_lab[:])
            extents_all.append(extents)
            logger.info(f"Median object size according to label: {extents}")

            slices = get_slices_list(image_raw.shape, patch_size, stride_sizes)
            # NOTE: crop images without filtering for non-empty slices
            if config['data']['slice']['filter_empty']:
                patches_raw, patches_lab = [], []
                for slice_idx in slices:
                    if len(np.unique(image_lab[slice_idx])) > 0:
                        patches_raw.append(image_raw[slice_idx])
                        patches_lab.append(image_lab[slice_idx])
            else:
                patches_raw = [image_raw[slice_idx] for slice_idx in slices]
                patches_lab = [image_lab[slice_idx] for slice_idx in slices]
            patches_raw_all.append(np.stack(patches_raw, axis=0))
            patches_lab_all.append(np.stack(patches_lab, axis=0))
        logger.debug("Success")
    return patches_raw_all, patches_lab_all, extents_all


def train_val_split(logger, X, Y):
    rng = np.random.RandomState(7)  # TODO: User input
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    logger.info(f'\n\t- patches:    {len(X):3d}\n\t- training:   {len(X_trn):3d}\n\t- validation: {len(X_val):3d}')
    return X_val, Y_val, X_trn, Y_trn


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.info("Running train.py")

    # Parse Arguments:
    args = utils.parse_arguments()

    # Load Config:
    config = utils.load_config(args.config_file)

    # TensorFlow Settings:
    utils.set_tf_op_parallelism_threads(config)

    # List Datasets:
    logger.info("Scanning datasets, please make sure file extensions are in lower case.")
    paths_dataset = config['data']['path']
    assert isinstance(paths_dataset, list)
    paths_file_dataset = utils.get_dataset_file_paths(paths_dataset)
    logger.info(f"Found the following files: \n\t{paths_file_dataset}")
    if paths_file_dataset == []:
        raise ValueError("Check your path for dataset, no dataset found.")

    # Load Datasets:
    logger.info("Loading datasets")
    patch_size = config['data']['slice']['patch_size']
    stride_sizes = config['data']['slice']['patch_size']
    patches_raw_all, patches_lab_all, extents_all = load_datasets(logger, config, paths_file_dataset, patch_size, stride_sizes)

    # Compute Overall Statistics:
    extents = np.mean(extents_all, axis=0)
    anisotropy = tuple(np.max(extents) / extents)
    config_stardist = config['stardist']
    grid = config_stardist.get('grid', tuple(1 if a > 1.5 else 2 for a in anisotropy))
    rays = Rays_GoldenSpiral(config_stardist['n_rays'], anisotropy=anisotropy)  # Rays on a Fibonacci lattice adjusted for anisotropy
    use_gpu = config_stardist['use_gpu'] and gputools_available()  # TODO: Restrict mory usage if gputools available

    logger.info(f"Extents is: {extents}")
    logger.info(f'Empirical anisotropy of labeled objects = {anisotropy}')
    logger.info(f"Predict on subsampled grid for increased efficiency and larger field of view: {grid}")
    logger.info(f"{'NOT ' if not use_gpu else ''}Using OpenCL-based computations for data generator during training (requires 'gputools')")

    # Build Model:
    model = StarDist3D(config=Config3D(
        rays=rays,
        grid=grid,
        anisotropy=anisotropy,
        use_gpu=use_gpu,
        n_channel_in=config_stardist['n_channel_in'],
        train_patch_size=patch_size,
        train_batch_size=config_stardist['train_batch_size'],
    ), name=config_stardist['model_name'], basedir=config_stardist['model_dir'])

    # Check Field of View:
    median_size = np.median(extents_all, axis=0)
    fov = np.array(model._axes_tile_overlap('ZYX'))

    logger.info(f"median object size:      {median_size}")
    logger.info(f"network field of view :  {fov}")
    if any(median_size > fov):
        logger.warning("WARNING: median object size larger than field of view of the neural network.")
    else:
        logger.info("Median object size smaller than field of view of the neural network.")

    # TODO: Normalisation from any data type to 0 ~ 1

    # Converting Datasets into Sequence
    logger.info("Converting datasets into raw/label patch sequences")
    patches_raw_all = np.vstack(patches_raw_all)
    patches_lab_all = np.vstack(patches_lab_all)
    X, Y = patches_raw_all, patches_lab_all
    if len(X) < 2:
        raise ValueError("Not enough training data.")

    # Train/val split
    logger.info("Splitting patches into train/validation sets")
    X_val, Y_val, X_trn, Y_trn = train_val_split(logger, X, Y)

    # Training
    logger.info("Training starting")
    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)
    logger.info("TRAIN SUCCESS!")
    model.optimize_thresholds(X_val, Y_val)
    logger.info("OPTIMISE SUCCESS!")
