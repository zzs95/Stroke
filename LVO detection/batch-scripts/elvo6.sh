#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=17g
#SBATCH --out=outfiles/elvo6.out
#SBATCH -t 120:00:00

conda activate pytorch_p37_nt
cd /home/ianpan/ufrc/elvo/src/
/home/ianpan/anaconda3/envs/pytorch_p37_nt/bin/python run.py configs/cv-fast50/i0/i0o6.yaml train --gpu 0 --num-workers 4