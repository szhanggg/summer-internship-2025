#!/bin/bash
#SBATCH --job-name=unet-training
#SBATCH --time=72:00:00

#SBATCH --nodes=2
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=512gb
module load anaconda
conda activate torch

srun python3 unettraining.py
