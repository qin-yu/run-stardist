# run-stardist
How to run StarDist? An example of using StarDist on plant datasets.

## Usage

Training:
```shell
$ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train_atul.yml
```

Prediction:
```shell
$ CUDA_VISIBLE_DEVICES=0 python predict.py --config configs/seg_atul.yml
```

## Setup Conda Environment for StarDist

```bash
$ conda env create -f environment.yml
$ conda activate stardist
```
