# Evaluating Robust Offline RL: A Reproduction of RIQL Under Diverse Data Corruption

COS 435: Reinforcement Learning Final Project. This repository is a clean working branch of our implementation of [Towards Robust Offline Reinforcement Learning under Diverse Data Corruption](https://arxiv.org/pdf/2310.12955).

This repository requires **Python 3.9** and **PyTorch 2.6 with CUDA 12.6**. 

## Setup Instructions

### 1. Conda Environment

Use Conda to create a new environment with Python 3.9:

```bash
conda create -n RIQL python=3.9
conda activate RIQL
```

We use pytorch=2.6 w/ CUDA 12.6
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### 3. Install python libraries
```bash
pip install -r requirements.txt
pip install --no-cache-dir git+https://github.com/Farama-Foundation/d4rl.git@master
```

#### 4. Install Mujoco
```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvzf mujoco210-linux-x86_64.tar.gz
```

#### 5. Setup env variables in .bashrc
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
```
#### 6. Install patchelf
```bash
conda install -c conda-forge patchelf
```
## Corrupt Data
To corrupt data, we use the data corruption scripts from the original paper. Running the corrupt_data.sh script saves all of the corrupted datasets locally to be used for training.
```
bash corrupt_data.sh
```
## Train RIQL
An example SLURM script for training RIQL can be found in [train-scripts](scripts/train-ant-actions.slurm). 