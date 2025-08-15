import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ..utils.data_pair_velo import FlowCpDataset 
from ..models.cvae import CVAE
from ..utils.Callback import TrainingLogger, make_dir, get_latest_dir
from ..utils.para_set import epochs, lr_p2z, batch_size_p2z, num_workers, epochs_p2z, alpha_end
from ..models.p2z import Pressure2Latent_new, Pressure2Latent_sha, Pressure2Latent_Light
import os 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load dataset
dataset_train = FlowCpDataset('/workspace/ljl/Junle_PIV_data/LatentFlow/data', mode='train')
dataset_val = FlowCpDataset('/workspace/ljl/Junle_PIV_data/LatentFlow/data', mode='val')
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size_p2z, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(dataset=dataset_val, batch_size=batch_size_p2z, shuffle=False, num_workers=num_workers)

# Load model
cvae = CVAE(latent_dim=128).to(device)
cvae.load_state_dict(torch.load('/workspace/ljl/Junle_PIV_data/LatentFlow/results/CVAE_1/checkpoint/checkpoint_epoch_200.pth')['model_state_dict'])
cvae.eval()
for p in cvae.parameters():
    p.requires_grad = False

# initialize Pressure2Latent model  
# Note: Pressure2Latent_new is a modified version of Pressure2Latent with additional parameters
p2z = Pressure2Latent_Light(input_dim=30, latent_dim=128, hidden_dim=256, dropout=0.2).to(device)
optimizer = optim.Adam(p2z.parameters(), lr=lr_p2z, weight_decay=1e-4)
loss_fn = nn.MSELoss()


def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    

# train p2z model
train_sample = 0
base_dir = get_latest_dir()  # Create a directory for saving results
checkpoint_path = os.path.join(base_dir, "end_end/checkpoint_p2z_light_new_loss")
os.makedirs(checkpoint_path, exist_ok=True)
best_path = os.path.join(base_dir, "best")
os.makedirs(best_path, exist_ok=True)
best_model_params = None
min_val_loss = float('inf')

def get_alpha(epoch, epochs, alpha_end):
    return alpha_end * (epoch / epochs) if epoch < epochs else alpha_end


# frozen cvae parameters
for param in cvae.parameters():
    param.requires_grad = False
    
log_file = os.path.join(checkpoint_path, 'train_log_p2z_new_loss.txt')
if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        f.write("Epoch\tTrain Loss\tVal Loss\n")

def kl_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    # D_KL( N(mu_q, var_q) || N(mu_p, var_p) )
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * torch.mean(
        (logvar_p - logvar_q) + (var_q + (mu_q - mu_p)**2) / var_p - 1
    )
for epoch in range(epochs_p2z):
    p2z.train()
    total_loss = 0
    #train_sample = 0
    alpha = get_alpha(epoch, epochs_p2z, alpha_end)
    
    for pressure, velocity in train_loader:
        pressure = pressure.to(device)
        velocity = velocity.to(device)
        
        with torch.no_grad():
            mu, logvar_cvae = cvae.encode(velocity, pressure)  # target latent

        pred_z, z_mu, logvar_p2z = p2z(pressure)
        #with torch.no_grad():
        pred_flow = cvae.decode(pred_z, pressure)  # predicted flow
        loss_train_mu = loss_fn(z_mu, mu)
        loss_train_flow = loss_fn(pred_flow, velocity)
        # KL divergence between predicted q(z|p) and N(0, I)
        kl_loss = kl_gaussians(z_mu, logvar_p2z, mu, logvar_cvae)
        loss_train = loss_train_mu + alpha* (loss_train_flow+ kl_loss)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        total_loss += loss_train.item()
        #train_sample += pressure.size(0)
        
    avg_loss_train = total_loss / len(train_loader)
    
    p2z.eval()
    total_val_loss = 0
    #sample_val_size =0
    with torch.no_grad():
        for pressure, velocity in test_loader:
            pressure = pressure.to(device)
            velocity = velocity.to(device)
            
            mu_cvae_val, logvar_cvae_val = cvae.encode(velocity, pressure)  # there is no gradient for mu as above line with torch.no_grad()
            pred_z_val, z_mu_val, logvar_p2z_val = p2z(pressure)
            
            pred_flow = cvae.decode(pred_z_val, pressure) # there is no gradient for pred_flow as above line with torch.no_grad()
            kl_loss_val = kl_gaussians(z_mu_val, logvar_p2z_val, mu_cvae_val, logvar_cvae_val)
            loss_val = loss_fn(pred_z_val, mu_cvae_val) + alpha * (loss_fn(pred_flow, velocity) + kl_loss_val)
            total_val_loss += loss_val.item()
            #sample_val_size += pressure.size(0)
            
    avg_loss_val = total_val_loss / len(test_loader)
    with open(log_file, 'a') as f:
        f.write(f"{epoch+1}\t{avg_loss_train:.6f}\t{avg_loss_val:.6f}\n")
    
    if avg_loss_val < min_val_loss:
        min_val_loss = avg_loss_val
        best_model_params = p2z.state_dict()
        
    if epoch % 200 == 0:
        save_checkpoint(p2z, optimizer, epoch, avg_loss_train, filename=f'{checkpoint_path}/p2z_checkpoint_epoch_{epoch}.pth')
        
    logger = TrainingLogger(print_every=10)
    if epoch % 10 == 0:
        logger.on_epoch_end(epoch=epoch, train_loss=avg_loss_train, current_loss_train=loss_train.item(), current_loss_val=loss_val.item(), val_loss=avg_loss_val)
    
if best_model_params is not None:
    torch.save(best_model_params, f'{checkpoint_path}/p2z_best_new_loss.pth')
    print(f"Best model saved with val_loss: {min_val_loss:.6f}")
