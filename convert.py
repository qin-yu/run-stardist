import tifffile
import h5py

with h5py.File("/home/qinyu/Datasets/AVijayan2021Ovules/1135_zoomed.h5", 'r') as f:
    print(f.keys())
    temp = f['raw'][:]
    print(f['raw'].dtype)

tifffile.imwrite("/home/qinyu/Datasets/AVijayan2021Ovules/1135_zoomed.tif", temp)
