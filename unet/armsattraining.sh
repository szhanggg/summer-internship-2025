#!/bin/bash
#SBATCH --job-name=sat-$1
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64gb
#SBATCH --partition=grace
module load miniforge
conda activate 3dclouddownstream

srun python3 satvisionpipeline.py $1 $2
