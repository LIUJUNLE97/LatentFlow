class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
##############################################################
import warnings
warnings.filterwarnings("ignore")

import torch 
import torch.nn as nn 
import numpy as np 
import math 
import os 
from datetime import datetime
import time 
from glob import glob
def MSELoss(preds,targs):
    return nn.MSELoss()(preds,targs)

class Fitter:
    
    def __init__(self, model, device, config, crop_size=224):
        self.config = config
        self.epoch = 0
        self.crop_size = crop_size

        self.base_dir = f'{os.path.dirname(os.getcwd())}/model_checkpoints/{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.log_path_save = f'{self.base_dir}/save_log.txt'
        # self.save_log_path = f'{self.base_dir}/save_log.txt'
        self.best_summary_loss = 10**5
        self.device = device
        self.model = model
        #self.model.to(self.device)
        #if torch.cuda.device_count() > 1:
        # print("Let's use",  f"GPU{device}!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            #self.model = nn.DataParallel(self.model)
        
        print("Let's use",  f"GPU: {device}!")

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.criterion = nn.MSELoss()
        self.metric = nn.L1Loss()
        self.log(f'Fitter prepared. Device is {self.device}')
        
        # self.iters_to_accumulate = 4 # gradient accumulation

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.now().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}') # 添加日志这一行去掉
            t = time.time()
            train_loss, train_maes, train_mape = self.train_one_epoch(train_loader)
    
            self.log(f'Train. Epoch: {self.epoch}, mse_loss: {train_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(f'Train. Epoch: {self.epoch}, mae: {train_maes.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(f'Train. Epoch: {self.epoch}, mape: {(train_mape.avg):.8f}, time: {(time.time() - t):.5f}')
            
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            # validation part
            val_loss, val_maes, val_mape = self.validation(validation_loader)
            with open(self.log_path_save, 'w') as f:
                log_str = f"{e+1}\t{train_loss:.4f}\t{val_loss:.4f}\n"
                f.write(log_str)

            self.log(f'Val. Epoch: {self.epoch}, val_loss: {val_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(f'Val. Epoch: {self.epoch}, val_mae: {val_maes.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(f'Val. Epoch: {self.epoch}, val_mape: {(val_mape.avg):.8f}, time: {(time.time() - t):.5f}')
            if val_loss.avg < self.best_summary_loss:
                self.best_summary_loss = val_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=val_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        val_loss = AverageMeter()
        val_maes = AverageMeter()
        val_mape = AverageMeter()
        t = time.time()
        for step, (frames, dmaps) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(val_loader)}, ' + \
                        f'mse_loss: {val_loss.avg:.8f}, ' + \
                        f'mae: {val_maes.avg:.8f}, ' + \
                        f'mape: {val_mape.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                batch_size, _, _, h, w = frames.shape
                #frames = frames.to(self.device)
                frames = frames.cuda().float()
                
                #dmaps = dmaps.to(self.device)
                dmaps = dmaps.cuda().float()
                
                # m, n = int(h//self.crop_size), int(w//self.crop_size)
                loss, preds_cnt, dmaps_cnt = 0, 0, 0
                preds = self.model(frames)
                loss = self.criterion(preds, dmaps)
                #preds_cnt = preds.cpu()
                #dmaps_cnt = dmaps.cpu()
                
                #preds_cnt += (preds.cpu()).sum()
                #dmaps_cnt += (dmaps.cpu()).sum()
                '''
                with torch.cuda.amp.autocast():
                    for i in range(m):
                        for j in range(n):
                            frame_patches = frames[:,:,:,self.crop_size*i:self.crop_size(i+1),self.crop_size*j:self.crop_size(j+1)]
                            dmaps_patches = dmaps[:,:,self.crop_size*i:self.crop_size(i+1),self.crop_size*j:self.crop_size(j+1)]
                            preds = self.model(frame_patches)
                            loss += self.criterion(preds, dmaps_patches)
                            preds_cnt += (preds.cpu()/LOG_PARA).sum()
                            dmaps_cnt += (dmaps_patches.cpu()/LOG_PARA).sum()
                '''

                val_loss.update(loss.detach().item(), batch_size)
                val_maes.update(self.compute_mae(preds, dmaps).detach().item())
                val_mape.update(self.compute_mape(preds, dmaps).detach().item())

        return val_loss, val_maes, val_mape
    
    def compute_mae(self, preds, targets):
        mae_loss = np.sum(np.abs(preds.cpu()-targets.cpu())/(np.prod(preds.shape)))
        return mae_loss
    
    def compute_mape(self, preds, targets):
        mape_loss = np.sum(np.abs(preds.cpu()-targets.cpu())/(targets.cpu()+1e-6)/(np.prod(preds.shape)))
        return mape_loss
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        print(self.model.encoder.blocks[0].norm1.weight.requires_grad)
        print(self.model.encoder.blocks[0].attn.qkv.weight.requires_grad)
        summary_loss = AverageMeter()
        maes = AverageMeter()
        mape = AverageMeter()
        t = time.time()
        for step, (frames, dmaps) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'mse: {summary_loss.avg:.8f}, ' + \
                        f'mae: {maes.avg:.8f}, ' + \
                        f'mape: {mape.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            batch_size = frames.shape[0]
            #frames = frames.to(self.device)
            frames = frames.cuda().float()
            
            #dmaps = dmaps.to(self.device)
            dmaps = dmaps.cuda().float()
            
            self.optimizer.zero_grad()
            #preds = self.model(frames.float())
            #loss = self.criterion(preds, dmaps)
            
            with torch.cuda.amp.autocast(): #native fp16
                #混合精度训练
                preds = self.model(frames.float())   
                loss = self.criterion(preds, dmaps)  # MSE loss 
                #preds_cnt = preds.cpu()
                #dmaps_cnt = dmaps.cpu()
                #preds_cnt = (preds.cpu()).sum()
                #dmaps_cnt = (dmaps.cpu()).sum()
            
            self.scaler.scale(loss).backward()
            # loss = loss / self.iters_to_accumulate # gradient accumulation

            
            summary_loss.update(loss.detach().item(), batch_size) # loss.item()
            maes.update(self.compute_mae(preds, dmaps).detach().item())  # MAE loss 
            mape.update(self.compute_mape(preds, dmaps).detach().item()) # MAPE loss
            #maes.update(abs(dmaps_cnt - preds_cnt))
            #mses.update((preds_cnt-dmaps_cnt)*(preds_cnt-dmaps_cnt))
            
            #self.optimizer.step()
            self.scaler.step(self.optimizer) # native fp16
            
            if self.config.step_scheduler:
                self.scheduler.step()
            
            self.scaler.update() #native fp16
            
#             if step == 10:
#                 break
                
                
#             if (step+1) % self.iters_to_accumulate == 0: # gradient accumulation

#                 self.optimizer.step()
#                 self.optimizer.zero_grad()

#                 if self.config.step_scheduler:
#                     self.scheduler.step()

        return summary_loss, maes, mape

    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
            #'amp': amp.state_dict() # apex
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')



def make_dir():
    import os
    dir = os.getcwd()

    folder = 'results'  
    base_dir = f'{dir}/{folder}/Unet'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        # log_path = f'{base_dir}/log.txt'
    else:
        i = 1
        while os.path.exists(f'{base_dir}_{i}'):
            i += 1
        base_dir = f'{base_dir}_{i}'
        os.makedirs(base_dir)
        # log_path = f'{base_dir}/log.txt'
    return base_dir
import torch 
import torch.nn.functional as F

def frequency_loss(pred, target):
    # 对时间维度做 FFT
    pred_fft = torch.fft.fft(pred, dim=2)
    target_fft = torch.fft.fft(target, dim=2)

    # 频域 MSE，计算复数模长的差异
    loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
    return loss

import torch
import torch.nn.functional as F

def frequency_loss_new(pred, target):
    """
    计算高于指定频率阈值部分的频域 MSE 损失（只针对时间维度 FFT）。
    
    参数:
        pred: [B, 1, T, S]，预测值（时间维度T=1000）
        target: [B, 1, T, S]，目标值
        sample_rate: 采样频率，例如 400Hz
        high_freq_threshold: 高频起始的频率阈值，例如 100Hz
    返回:
        高频频率段的 MSE 损失（频域模值）
    """
    # 时间维度长度
    sample_rate=400
    high_freq_threshold=100
    T = pred.size(2)
    freq_resolution = sample_rate / T  # 每一点代表的频率

    # 起始索引位置
    cutoff_idx = int(high_freq_threshold / freq_resolution)

    # FFT，按时间维度，输出 [B, 1, T, S]
    pred_fft = torch.fft.fft(pred, dim=2)
    target_fft = torch.fft.fft(target, dim=2)

    # 频谱是对称的，取前半部分（实数信号）
    fft_len = T // 2 + 1
    pred_fft = pred_fft[:, :, :fft_len, :]
    target_fft = target_fft[:, :, :fft_len, :]

    # 高频裁剪
    pred_high = pred_fft[:, :, cutoff_idx:, :]
    target_high = target_fft[:, :, cutoff_idx:, :]

    # 计算模值的 MSE（也可替换为 real/imag 分别 MSE）
    pred_mag = torch.abs(pred_high)
    target_mag = torch.abs(target_high)

    loss = F.mse_loss(pred_mag, target_mag)
    return loss

def train_model(model, train_dataloader, val_dataloader, num_epochs, beta, optimizer, base_dir, resume_path):
    import torch
    import torch.nn as nn
    import os

    checkpoint_path = os.path.join(base_dir, "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)
    best_path = os.path.join(base_dir, "best")
    os.makedirs(best_path, exist_ok=True)

    # 初始化参数
    start_epoch = 0
    min_val_loss = float('inf')
    best_model_params = None

    # 如果有 checkpoint，加载
    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint.get('min_val_loss', float('inf'))
        print(f"Resumed training from epoch {start_epoch}, min_val_loss: {min_val_loss:.4f}")

    train_criterion = nn.MSELoss()
    val_criterion = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 如果训练日志文件不存在，先写表头
    log_file = os.path.join(base_dir, 'train_log.txt')
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Epoch\tTrain Loss\tVal Loss\n")

    try:
        for epoch in range(start_epoch, num_epochs):
            dynamic_beta = beta * (epoch + 1) / num_epochs

            model.train()
            train_loss = 0.0
            total_samples = 0

            for input_data, target_data in train_dataloader:
                optimizer.zero_grad()
                inputs = input_data.unsqueeze(1).float().to(device)
                targets = target_data.unsqueeze(1).float().to(device)

                outputs = model(inputs)
                mse_loss_train = train_criterion(outputs, targets)
                loss = mse_loss_train + dynamic_beta * frequency_loss_new(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

            train_loss /= total_samples

            model.eval()
            val_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for input_data, target_data in val_dataloader:
                    inputs = input_data.unsqueeze(1).float().to(device)
                    targets = target_data.unsqueeze(1).float().to(device)

                    outputs = model(inputs)
                    mse_loss_val = val_criterion(outputs, targets)
                    loss = mse_loss_val + dynamic_beta * frequency_loss_new(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    total_val_samples += inputs.size(0)

            val_loss /= total_val_samples

            # 写入日志文件（追加模式）
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_loss:.4f}\n")

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model_params = model.state_dict()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if (epoch + 1) % 50 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'min_val_loss': min_val_loss,
                }
                torch.save(checkpoint, f'{checkpoint_path}/checkpoint_epoch_{epoch+1}.pth')

    except KeyboardInterrupt:
        print("Training interrupted. Saving current best model...")

    # 训练结束后保存最佳模型
    if best_model_params is not None:
        torch.save(best_model_params, f'{best_path}/unet.pth')
        print(f"Best model saved with val_loss: {min_val_loss:.4f}")