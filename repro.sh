#!/bin/bash
#SBATCH --gpus=4 
source slurm_config.sh
#SBATCH -A ${SLURM_ACCOUNT}
#SBATCH -t 10:00:00
#SBATCH --reservation safe
#SBATCH hetjob
#SBATCH -p berzelius-cpu
#SBATCH -n1
#SBATCH -c12
#SBATCH -A ${SLURM_ACCOUNT}
#SBATCH -t 05:00:00
source .venv/bin/activate
export HF_HOME=/proj/berzelius-2025-303/users/x_gabdu/random/hf_cache
srun --het-group=0 dvc repro books_to_paragraphs paragraphs_to_chunks chunks_to_embeddings
srun --het-group=1 dvc repro embeddings_to_qdrant
