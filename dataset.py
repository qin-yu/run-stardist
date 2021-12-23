import numpy as np
import pandas as pd
import h5py
import glob
from pprint import pprint
from skimage.measure import regionprops
import time
import matplotlib.pyplot as plt


def check_dataset_names(path_file):
    with h5py.File(path_file, 'r') as f:
        print(f.keys())
    return


def get_files_from_folder(path_folder, ext='h5'):
    pathlist_files = glob.glob(path_folder + '/*.' + ext)
    pprint(pathlist_files)
    return(pathlist_files)


def file_properties(path_file):
    with h5py.File(path_file, 'r') as f:
        if 'label' not in f.keys():
            message_err = "'label' dataset must be included in the input HDF5 file! " + \
                          f"Only found the following keys:\n{f.keys()}"
            raise KeyError(message_err)

        time_start = time.time()
        dset = f['label'][:]
        time_end = time.time()
        print(f"Load time: {time_end - time_start}")

        regs = regionprops(dset)

        # properties = [dset.shape, dset.dtype, ]

    return regs


def understand_dataset(path_folder, ext='h5'):
    pathlist_files = get_files_from_folder(path_folder, ext='h5')
    for path_file in pathlist_files:
        file_properties(path_file)
    return


if __name__ == '__main__':
    path_folder = "/g/kreshuk/yu/Datasets/TMody2021Ovules/train"
    pathlist_files = get_files_from_folder(path_folder, ext='h5')
    # understand_dataset(path_folder)
    regs = file_properties(pathlist_files[0])
    extents = np.array([np.array(r.bbox[3:]) - np.array(r.bbox[:3]) for r in regs])
    areas = np.array([r.area for r in regs])
    plt.hist(areas, bin=50)
    print(regs[0].bbox, regs[0].area)
