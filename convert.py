from ast import Try
import os
import h5py
import glob
import tqdm
import logging
import argparse
import tifffile

import utils


def save_tiff(dest, data, dtype):
    tifffile.imwrite(dest, data=data.astype(dtype), imagej=True)


def read_hdf5(path, name):
    with h5py.File(path, 'r') as f:
        return f[name][:]


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.info("Parsing arguments")
    parser = argparse.ArgumentParser(description="Parse Arguments!")
    parser.add_argument('-n', dest='dset_name', required=True)
    parser.add_argument('-t', dest='dset_type', required=False, default='float32')
    parser.add_argument('-i', dest='path_i', required=True)
    parser.add_argument('-o', dest='path_o', required=True)
    args = parser.parse_args()
    logger.info(f"Arguments are: {args}")

    if os.path.isfile(args.path_i) and os.path.isfile(args.path_o):
        save_tiff(args.path_o, read_hdf5(args.path_i, args.dset_name), args.dset_type)
    elif os.path.isdir(args.path_i) and os.path.isdir(args.path_o):
        path_list = glob.glob(args.path_i + '/*.h5')
        for path_i in tqdm.tqdm(path_list):
            file_name = os.path.splitext(os.path.basename(path_i))[0]
            path_o = f"{args.path_o}/{file_name}_{args.dset_name}.tif"
            try:
                save_tiff(path_o, read_hdf5(path_i, args.dset_name), args.dset_type)
            except KeyError:
                print(f"WARNNING: No {args.dset_name} in {path_i}")
            
    else:
        raise ValueError("Input and output path should both be files or both be folders.")