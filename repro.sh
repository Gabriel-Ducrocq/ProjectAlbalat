#!/bin/bash
#SBATCH --gpus=4
source slurm_config.sh
#SBATCH -A ${SLURM_ACCOUNT} 
#SBATCH -t 10:00:00
#SBATCH --reservation safe
source .venv/bin/activate
export HF_HOME=/proj/berzelius-2025-303/users/x_gabdu/random/hf_cache
dvc repro
