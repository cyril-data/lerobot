#!/bin/bash

#SBATCH -A hir@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=19:00:00

module load miniforge/24.11.3
conda run -n lerobot python lerobot/scripts/train.py   --dataset.repo_id="run"   --policy.type=act   --output_dir=outputs/train/act_so101_test   --job_name=act_so101_test   --policy.device=cuda   --wandb.enable=Fals
e --dataset.root "data"
