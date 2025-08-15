import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.cvae import CVAE

# from models.p2z import Pressure2Latent_new
import numpy as np 

input = np.load('/workspace/ljl/Junle_PIV_data/LatentFlow/data/Cp_30000.npy')[:10240, :]
input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Load model
cvae = CVAE(latent_dim=128).to(device)
cvae.load_state_dict(torch.load('/workspace/ljl/Junle_PIV_data/LatentFlow/results/CVAE_1/checkpoint/checkpoint_epoch_200.pth')['model_state_dict'])
cvae.eval()
for p in cvae.parameters():
    p.requires_grad = False

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)  

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        out = out + residual  
        out = self.norm(out)
        return out

class Pressure2Latent_new(nn.Module):
    def __init__(self, input_dim=30, latent_dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # output
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout)
        )
        # output
        self.output_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, p):
        x = self.input_layer(p)
        x = self.res_blocks(x)
        z = self.output_layer(x)
        return z
# initialize Pressure2Latent model
p2z = Pressure2Latent_new(input_dim=30, latent_dim=128, hidden_dim=256, dropout=0.1).to(device)
p2z.load_state_dict(torch.load('/workspace/ljl/Junle_PIV_data/LatentFlow/results/CVAE_1/end_end/checkpoint_p2z/p2z_checkpoint_epoch_200.pth')['model_state_dict'])
p2z.eval()
for p in p2z.parameters():
    p.requires_grad = False

reconstructed_results = []
for i in range(10240):
    pressure = input[:, i, :].to(device)
    z_pred = p2z(pressure)
    pred_flow = cvae.decode(z_pred, pressure)
    reconstructed_results.append(pred_flow.cpu().numpy())
    
reconstructed_results = np.stack(reconstructed_results, axis=0)
np.save('/workspace/ljl/Junle_PIV_data/LatentFlow/results/CVAE_1/end_end/checkpoint_p2z/infer_results_longer.npy', reconstructed_results)