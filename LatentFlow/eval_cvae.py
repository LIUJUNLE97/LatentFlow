import torch
import torch.nn as nn
import torch.optim as optim
import os 
from torch.utils.data import DataLoader
from utils.data_pair_velo import FlowCpDataset 
from models.cvae import CVAE
from utils.Callback import TrainingLogger, make_dir
from utils.para_set import epochs, lr, batch_size, num_workers, beta_end
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load dataset
dataset_train = FlowCpDataset('/workspace/ljl/Junle_PIV_data/LatentFlow/data', mode='train')
dataset_val = FlowCpDataset('/workspace/ljl/Junle_PIV_data/LatentFlow/data', mode='val')
train_loader = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=num_workers)

cvae = CVAE(latent_dim=128).to(device)
cvae.load_state_dict(torch.load('results/CVAE_1/checkpoint/checkpoint_epoch_200.pth')['model_state_dict'])
cvae.eval()
reconstcuted_results = []
ground_truth_velo = []
loss_save = []
eval_loss = nn.MSELoss()
with torch.no_grad():
    for pressure, velocity in test_loader:
        pressure = pressure.to(device)
        velocity = velocity.to(device)
        # mu, _ = cvae.encode(velocity, pressure)
        recon_x, mu, logvar = cvae(velocity, pressure)
        loss_eval = eval_loss(recon_x, velocity)
        loss_save.append(loss_eval.item())
        reconstcuted_results.append(recon_x.cpu().numpy())
        ground_truth_velo.append(velocity.cpu().numpy())

reconstcuted_results=np.stack(reconstcuted_results, axis=0)
ground_truth_velo=np.stack(ground_truth_velo, axis=0)

np.save('results/CVAE_1/reconstcuted_results_cvae.npy', reconstcuted_results)
np.save('results/CVAE_1/loss_save_cvae.npy', loss_save)
np.save('results/CVAE_1/ground_truth_velo_cvae.npy', ground_truth_velo)