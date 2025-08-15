import numpy as np
import torch
from torch.utils.data import Dataset

class UDataset(Dataset):
    # This is used to smaple the velocity from the dataset for VAE
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.data[idx, :, :]
        output_data = self.data[idx, :, :]
        
        return input_data, output_data
    
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class FlowCpDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        """
        data_dir: data folder, which includes Cp_1200.npy & velocities_normalized.npy
        mode: 'train' / 'val', mutual selection
        """
        self.cp = np.load(os.path.join(data_dir, "Cp_1200.npy"))           # (1200, 30)
        self.vel = np.load(os.path.join(data_dir, "velocities_normalized.npy"))  # (1200, 2, H, W)

        N = self.cp.shape[0]
        if mode == 'train':
            self.cp = self.cp[:int(0.9*N)]
            self.vel = self.vel[:int(0.9*N)]
        else:
            self.cp = self.cp[int(0.9*N):]
            self.vel = self.vel[int(0.9*N):]

    def __len__(self):
        return len(self.cp)

    def __getitem__(self, idx):
        cp_tensor = torch.tensor(self.cp[idx], dtype=torch.float32)         # (30,)
        vel_tensor = torch.tensor(self.vel[idx], dtype=torch.float32)       # (2, H, W)
        return cp_tensor, vel_tensor
