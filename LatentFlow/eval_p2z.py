import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data_pair_velo import FlowCpDataset 
from models.cvae import CVAE
from utils.Callback import TrainingLogger, make_dir, get_latest_dir
from utils.para_set import epochs, lr_p2z, batch_size, num_workers, epochs_p2z
from models.p2z import Pressure2Latent_new, Pressure2Latent_sha, Pressure2Latent_Light
import os 
import numpy as np 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load dataset
dataset_train = FlowCpDataset('/workspace/ljl/Junle_PIV_data/LatentFlow/data', mode='train')
dataset_val = FlowCpDataset('/workspace/ljl/Junle_PIV_data/LatentFlow/data', mode='val')
train_loader = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=num_workers)

# Load model
cvae = CVAE(latent_dim=128).to(device)
cvae.load_state_dict(torch.load('/workspace/ljl/Junle_PIV_data/LatentFlow/results/CVAE_1/checkpoint/checkpoint_epoch_200.pth')['model_state_dict'])
cvae.eval()
for p in cvae.parameters():
    p.requires_grad = False

# initialize Pressure2Latent model
p2z = Pressure2Latent_Light(input_dim=30, latent_dim=128, hidden_dim=256, dropout=0.2).to(device)
p2z.load_state_dict(torch.load('//workspace/ljl/Junle_PIV_data/LatentFlow/results/CVAE_1/end_end/checkpoint_p2z_light_new_loss/p2z_best_new_loss.pth'))
z_cvae = []
z_p2z = []
loss_save = []
loss_fn = nn.MSELoss()
reconstructed_results = []
for pressure, velocity in test_loader:
    velocity = velocity.to(device)
    pressure = pressure.to(device)
    #print(pressure.shape)
    with torch.no_grad():
        mu, logvar = cvae.encode(velocity, pressure)
        z = cvae.reparameterize(mu, logvar)
        z_cvae.append(z.cpu().numpy())
        p2z.eval()
        z_pred, _, _ = p2z(pressure)
        z_p2z.append(z_pred.cpu().numpy())
        loss = loss_fn(z_pred, mu)
        loss_save.append(loss.item())
        reconstructed = cvae.decode(z_pred, pressure)
        reconstructed_results.append(reconstructed.cpu().numpy())
        
z_cvae = np.stack(z_cvae, axis=0)
z_p2z = np.stack(z_p2z, axis=0)
loss_save = np.array(loss_save)
reconstructed_results = np.stack(reconstructed_results, axis=0)
np.save('/workspace/ljl/Junle_PIV_data/LatentFlow/results/results_compre/new_loss/z_cvae_sha.npy', z_cvae)
np.save('/workspace/ljl/Junle_PIV_data/LatentFlow/results/results_compre/new_loss/z_p2z_sha.npy', z_p2z)
np.save('/workspace/ljl/Junle_PIV_data/LatentFlow/results/results_compre/new_loss/loss_save_p2z_sha.npy', loss_save)
np.save('/workspace/ljl/Junle_PIV_data/LatentFlow/results/results_compre/new_loss/reconstructed_results_p2z_sha.npy', reconstructed_results)