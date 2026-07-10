#!/bin/bash
#SBATCH --gpus=4
#SBATCH -A ${SLURM_ACCOUNT} 
#SBATCH -t 10:00:00
source .venv/bin/activate
source slurm_config.sh
dvc repro
