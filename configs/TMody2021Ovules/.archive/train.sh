#!/bin/bash

#SBATCH -A kreshuk
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem 600G
#SBATCH -p gpu
#SBATCH -C "gpu=A100"
#SBATCH --gres=gpu:1
#SBATCH -t 120:00:00
#SBATCH -o /g/kreshuk/yu/.lognohup/TMody2021Ovules_training_2.out
#SBATCH -e /g/kreshuk/yu/.lognohup/TMody2021Ovules_training_2.err
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=qin.yu@embl.de

source ~/.bashrc
conda activate stardist
python /g/kreshuk/yu/run-stardist-tmp/train.py --config_file /g/kreshuk/yu/run-stardist-tmp/configs/TMody2021Ovules/train_slurm.yml
