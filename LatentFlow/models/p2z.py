import torch
import torch.nn as nn
import torch.nn.functional as F

class Pressure2Latent(nn.Module):
    def __init__(self, input_dim=30, latent_dim=128):
        super(Pressure2Latent, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, p):
        return self.model(p)
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            ResidualBlock(hidden_dim, dropout)
        )
        # output
        self.output_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, p):
        x = self.input_layer(p)
        x = self.res_blocks(x)
        z = self.output_layer(x)
        return z
class Pressure2Latent_sha(nn.Module):
    
    def __init__(self, input_dim=30, latent_dim=128, hidden_dim=256, dropout=0.2):
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
    
class Pressure2Latent_Light(nn.Module):
    def __init__(self, input_dim=30, latent_dim=128, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res = ResidualBlock(hidden_dim, dropout)  # 只保留一个残差块，模型更浅
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, p, sample: bool = True, deterministic_eval: bool = True):
        """
        sample=True and self.training: z
        OR: z=mu
        deterministic_eval=True  eval(): mu
        """
        h = self.res(self.feat(p))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        use_sampling = sample and self.training
        if deterministic_eval and (not self.training):
            use_sampling = False

        z = self.reparameterize(mu, logvar) if use_sampling else mu
        return z, mu, logvar