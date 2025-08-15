import torch
import torch.nn as nn
import torch.optim as optim
import os 
from torch.utils.data import DataLoader
from ..utils.data_pair_velo import FlowCpDataset 
from ..models.cvae import CVAE
from ..utils.Callback import TrainingLogger, make_dir
from ..utils.para_set import epochs, lr, batch_size, num_workers, beta_end

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load dataset
dataset_train = FlowCpDataset('/workspace/ljl/Junle_PIV_data/LatentFlow/data', mode='train')
dataset_val = FlowCpDataset('/workspace/ljl/Junle_PIV_data/LatentFlow/data', mode='val')
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

cvae = CVAE(latent_dim=128).to(device)
optimizer = optim.Adam(cvae.parameters(), lr=lr)
recon_loss_fn = nn.MSELoss()

def loss_function(recon_x, x, mu, logvar, beta):
    recon_loss = recon_loss_fn(recon_x, x)
    kld = -beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld


def save_checkpoint(model, optimizer, epoch, loss, min_loss, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'min_loss': min_loss
    }
    torch.save(checkpoint, filename)
    
def get_beta(epoch, epochs, beta_end):
    return beta_end * (epoch / epochs) if epoch < epochs else beta_end

base_dir = make_dir()  # Create a directory for saving results
checkpoint_path = os.path.join(base_dir, "checkpoint")
os.makedirs(checkpoint_path, exist_ok=True)
best_path = os.path.join(base_dir, "best")
os.makedirs(best_path, exist_ok=True)
best_model_params = None
# train the model
for epoch in range(epochs):
    cvae.train()
    total_loss = 0
    total_train_samples = 0
    min_val_loss = float('inf')
    log_file = os.path.join(base_dir, 'train_log.txt')
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Epoch\tTrain Loss\tVal Loss\n")
    
    for pressure, velocity in train_loader:
        pressure = pressure.to(device)  # torch.size[b, 30]  -> [b, 1, 30]
        velocity = velocity.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = cvae(velocity, pressure)
        beta = get_beta(epoch, epochs, beta_end=beta_end)
        loss_train = loss_function(recon_x, velocity, mu, logvar, beta)
        loss_train.backward()
        optimizer.step()
        total_loss += loss_train.item()
        total_train_samples = velocity.size(0) if total_train_samples == 0 else total_train_samples + velocity.size(0)
        
    avg_loss = total_loss / total_train_samples
    total_val_loss = 0
    total_val_samples = 0
    cvae.eval()
    with torch.no_grad():
        for pressure, velocity in test_loader:
            pressure = pressure.to(device)
            velocity = velocity.to(device)
            recon_x, mu, logvar = cvae(velocity, pressure)
            beta = get_beta(epoch, epochs, beta_end=beta_end)
            loss_val = loss_function(recon_x, velocity, mu, logvar, beta)
            total_val_loss += loss_val.item()
            total_val_samples += velocity.size(0)
            
    avg_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0
    
    with open(log_file, 'a') as f:
        f.write(f"{epoch+1}\t{avg_loss:.4f}\t{avg_val_loss:.4f}\n")

    if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_model_params = cvae.state_dict()
    
    if epoch % 200 == 0:
        save_checkpoint(cvae, optimizer, epoch, avg_loss, filename=f'{checkpoint_path}/checkpoint_epoch_{epoch}.pth', min_loss=min_val_loss)
    logger = TrainingLogger(print_every=10)

    logger.on_epoch_end(epoch=epoch, train_loss=avg_loss, current_loss_train=loss_train.item(), current_loss_val=loss_val.item(), val_loss=avg_loss)  # Assuming no validation loss for now

# save the trained model
    if best_model_params is not None:
        torch.save(best_model_params, f'{best_path}/cvae_best.pth')
        print(f"Best model saved with val_loss: {min_val_loss:.4f}")
