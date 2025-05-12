# dataset.py

import torch
from torch.utils.data import Dataset
import h5py

class Dataset(Dataset):
    def __init__(self, path):
        with h5py.File(path, 'r') as f:
            self.data = {key: torch.from_numpy(f[key][:]).float() for key in f.keys() if isinstance(f[key], h5py.Dataset)}
        self.state_dim = self.data['observations'].shape[1]
        self.action_dim = self.data['actions'].shape[1]
        # self.max_action = self.data['actions'].max()     
        self.size = self.data['observations'].shape[0]   
        
        # compute normalization observations and next_observations
        concatenated_obs = torch.cat((self.data['observations'], self.data['next_observations']), dim=0)
        self.mean = concatenated_obs.mean(axis=0)
        self.std = torch.clamp(concatenated_obs.std(axis=0), min=1e-6)
        self.data['observations'] = (self.data['observations'] - self.mean) / self.std
        self.data['next_observations'] = (self.data['next_observations'] - self.mean) / self.std
        
        print(f"Loaded dataset of size: {self.size}")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return (
            self.data['observations'][idx, :],       
            self.data['actions'][idx, :],    
            self.data['rewards'][idx],     
            self.data['next_observations'][idx, :],  
            self.data['terminals'][idx],
        )
    def obs(self):
        return self.data['observations']
    
    def next_obs(self):
        return self.data['next_observations']
    
    def actions(self):
        return self.data['actions']
    
    def rewards(self):
        return self.data['rewards']
    
if __name__ == "__main__":
    dataset = Dataset('hopper-datasets/hopper-medium-replay-v2-corrupt-acts.hdf5')
    print(dataset.mean, dataset.std)                      # no zeros or NaNs
    print((torch.isnan(dataset.obs()).any(), torch.isnan(dataset.next_obs()).any()))
    print((torch.isnan(dataset.actions()).any(), torch.isnan(dataset.rewards()).any()))
    print(dataset.__len__())
    