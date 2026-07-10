#!/bin/bash
#SBATCH --gpus=1 
#SBATCH -A berzelius-2025-303 
#SBATCH -t 24:00:00
#BATCH --reservation safe
module load Mambaforge
#conda activate cryoSPHERE_pose_estimation
conda activate /proj/berzelius-2025-303/users/x_gabdu/cryosphere061_dev
cryosphere_train --experiment_yaml parameters_mcl.yaml
