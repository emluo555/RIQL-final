## Example slurm script for training RIQL on adroit cluster

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=50G
#SBATCH --output=/scratch/network/{netid}/RIQL/slrm_log/train_%j.out
#SBATCH --error=/scratch/network/{netid}/RIQL/slrm_log/train_%j.err
#SBATCH --mem=50G
#SBATCH --partition=gpu
source ~/.bashrc
conda activate RIQL
cd /scratch/network/{netid}/RIQL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/RIQL/include
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export MUJOCO_PATH=$HOME/.mujoco/mujoco210
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco210/mjkey.txt
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8   
export CPATH=$CONDA_PREFIX/include

python RIQL/train_RIQL.py \
    --num_workers 8 \
    --save_folder checkpoints/deterministic_ant_actions \
    --env ant-datasets/ant-medium-replay-v2-corrupt-acts.hdf5 \
    --env_name Ant-v2 \
    --K 3 \
    --alpha 0.25 \
    --delta 0.5