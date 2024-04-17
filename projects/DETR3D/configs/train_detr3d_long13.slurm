#!/bin/bash

#SBATCH --job-name=train
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhong_l@sjtu.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load cuda/11.3.1
module load miniconda3
source activate mmlab
cd $SLURM_SUBMIT_DIR
./tools/dist_train.sh projects/DETR3D/configs/detr3d_r101_gridmask_m2dset_l13.py 2
