import os
import glob
import h5py
import tifffile
import numpy as np
from tqdm import tqdm
from pprint import pprint
from natsort import natsorted
from pathlib import Path
# from skimage.transform import rescale
from scipy.ndimage import zoom

from utils import compute_grid


def combine_tiff_to_h5(output_folder_path, raw_path, lab_path, file_id):
    raw = tifffile.imread(raw_path)
    lab = tifffile.imread(lab_path)
    with h5py.File(output_folder_path + f"{file_id}.h5", 'w') as f:
        f.create_dataset("raw",   data=zoom(raw, (1., .5, .5), order=1), compression='gzip')
        f.create_dataset("label", data=zoom(np.flip(lab, axis=1), (1., .5, .5), order=0), compression='gzip')


def convert_all_tiff(output_folder_path, raw_file_list, lab_file_list):
    for raw_path, lab_path in zip(raw_file_list, tqdm(lab_file_list)):
        file_id = os.path.basename(raw_path)[:4]
        combine_tiff_to_h5(output_folder_path, raw_path, lab_path, file_id)


input_folder_path = "/g/kreshuk/yu/Datasets/TMody2021Ovules/.original/"
output_folder_path = "/g/kreshuk/yu/Datasets/TMody2021Ovules/train_xyds2/"
Path(output_folder_path).mkdir(parents=True, exist_ok=True)

raw_file_list = natsorted(glob.glob(input_folder_path + "*_n_stain.tif"))
lab_file_list = natsorted(glob.glob(input_folder_path + "*_n_stain_segmented_corrected.tif"))
pprint(raw_file_list)
pprint(lab_file_list)

convert_all_tiff(output_folder_path, raw_file_list, lab_file_list)

for raw_path, lab_path in zip(raw_file_list, tqdm(lab_file_list)):
    file_id = os.path.basename(raw_path)[:4]
    print(compute_grid(file_path=output_folder_path + f"{file_id}.h5"))
