# Run Stardist <!-- omit in toc --> 
A complete training and inference pipeline of StarDist with an example on plant datasets.

- [Usage](#usage)
- [Installation](#installation)
## Usage

Training:
```shell
$ CUDA_VISIBLE_DEVICES=7 python train.py --config configs/AVijayan2021Ovules/train_athul.yml 
```

Prediction:
```shell
$ CUDA_VISIBLE_DEVICES=7 python predict.py --config configs/TMody2021Ovules/seg_teja.yml
```

## Installation

```bash
$ conda env create -f environment.yml
$ conda activate stardist
```
