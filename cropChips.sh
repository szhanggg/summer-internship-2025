#!/bin/bash
module load anaconda
conda activate ilab-pytorch

python3 cropChips.py $1
