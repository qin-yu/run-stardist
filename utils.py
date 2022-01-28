import argparse
import logging
import os
import h5py
import yaml
import numpy as np
from glob import glob
from natsort import natsorted

from stardist import calculate_extents

EXT_HDF5 = ['hdf5', 'h5', '.hdf5', '.h5']
EXT_TIFF = ['tiff', 'tif', '.tiff', '.tif']


def get_gpu_info():
    logger = logging.getLogger(__name__)
    logger.info(f"Using GPU:{os.environ['CUDA_VISIBLE_DEVICES']} on {os.uname().nodename}")
    logger.info(f"Using CONDA environment '{os.environ['CONDA_DEFAULT_ENV']}' at {os.environ['CONDA_PREFIX']}")


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


def is_hdf5(string):
    s = string.lower()
    return s in EXT_HDF5


def is_tiff(string):
    s = string.lower()
    return s in EXT_TIFF


def tiff_path(string):
    if file_path(string):  # File must exist
        _, file_extension = os.path.splitext(string)
        print(file_extension)
        if is_tiff(file_extension):
            return True
    raise FileNotFoundError("Not a TIFF file.")


def hdf5_path(string):
    if file_path(string):  # File must exist
        _, file_extension = os.path.splitext(string)
        print(file_extension)

        if is_hdf5(file_extension):
            return True
    raise FileNotFoundError("Not a HDF5 file.")


def parse_arguments():
    logger = logging.getLogger(__name__)
    logger.info("Parsing arguments")
    parser = argparse.ArgumentParser(description="Parse Arguments!")
    parser.add_argument('--config_file', dest='config_file', required=True, type=file_path)
    args = parser.parse_args()
    logger.info(f"Arguments are: {args}")
    return args


def load_config(path_config):
    logger = logging.getLogger(__name__)
    logger.info("Loading config file")
    config = yaml.safe_load(open(path_config, 'r'))
    config_str = '    ' + yaml.dump(config, indent=4).replace('\n', '\n    ')
    logger.info(f"Config contains: \n{config_str}")
    return config


def set_tf_op_parallelism_threads(config):
    import tensorflow as tf
    logger = logging.getLogger(__name__)
    # gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    # for device in gpu_devices:
    #     tf.config.experimental.set_memory_growth(device, True)
    try:
        config_tensorflow = config['tensorflow']
        config_threading = config_tensorflow['threading']
        logger.debug(f"inter_op_parallelism_threads = {tf.config.threading.get_inter_op_parallelism_threads()}")
        logger.debug(f"intra_op_parallelism_threads = {tf.config.threading.get_intra_op_parallelism_threads()}")
        tf.config.threading.set_inter_op_parallelism_threads(config_threading['inter_op_parallelism_threads'])
        tf.config.threading.set_intra_op_parallelism_threads(config_threading['intra_op_parallelism_threads'])
    except KeyError as exc:
        logger.info("No change to TensorFlow's threading settings")
    logger.debug(f"inter_op_parallelism_threads = {tf.config.threading.get_inter_op_parallelism_threads()}")
    logger.debug(f"intra_op_parallelism_threads = {tf.config.threading.get_intra_op_parallelism_threads()}")


def get_dataset_file_paths(paths_dataset, format='hdf5'):
    paths_file_dataset = []
    for path_dataset in paths_dataset:
        if os.path.isfile(path_dataset):
            paths_file_dataset.append(path_dataset)
        elif os.path.isdir(path_dataset):
            if is_hdf5(format):
                for file_extension in EXT_HDF5:  # `glob` ignores multiple `/` and only support shell-style wildcards.
                    paths_file_dataset += natsorted(glob(path_dataset + "/*." + file_extension))
            elif is_tiff(format):
                for file_extension in EXT_TIFF:
                    paths_file_dataset += natsorted(glob(path_dataset + "/*." + file_extension))

    return paths_file_dataset


def read_h5_voxel_size(file_path, name):
    with h5py.File(file_path, "r") as f:
        ds = f[name]

        # parse voxel_size
        if 'element_size_um' in ds.attrs:
            return ds.attrs['element_size_um']
        else:
            return None


def compute_grid(file_path, dataset_name='label'):
    with h5py.File(file_path, 'r') as f:
        Y = f[dataset_name][:]
    extents = calculate_extents(Y)
    anisotropy = tuple(np.max(extents) / extents)
    print('empirical anisotropy of labeled objects = %s' % str(anisotropy))
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)
    return grid
