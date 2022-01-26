import os
import glob
import h5py
import tifffile
from tqdm import tqdm
from pprint import pprint
from natsort import natsorted
from pathlib import Path
from scipy.ndimage import zoom


def combine_tiff_to_h5(output_folder_path, raw_path, file_id, rescale_factors):
    raw = tifffile.imread(raw_path)
    with h5py.File(output_folder_path + f"{file_id}.h5", 'w') as f:
        f.create_dataset("raw",   data=zoom(raw, rescale_factors, order=1), compression='gzip')


def convert_all_tiff(output_folder_path, raw_file_list, rescale_factors):
    for raw_path in tqdm(raw_file_list):
        file_id = os.path.basename(raw_path)[:4]
        combine_tiff_to_h5(output_folder_path, raw_path, file_id, rescale_factors)


input_folder_path = "/g/kreshuk/yu/Datasets/TMody2021Ovules/.original_new/"
output_folder_path = "/g/kreshuk/yu/Datasets/TMody2021Ovules/test_xyds2/"
Path(output_folder_path).mkdir(parents=True, exist_ok=True)

raw_file_list = natsorted(glob.glob(input_folder_path + "*_nstain.tif"))
pprint(raw_file_list)

convert_all_tiff(output_folder_path, raw_file_list, (1., .5, .5))

