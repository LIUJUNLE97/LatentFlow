import torch
import torch.nn as nn
import torch.nn.functional as F
'''
'''
class CVAE_old(nn.Module):
    def __init__(self, latent_dim=128):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_channels = 2  # u, v. I have not used vorticity in the model, but it can be added if needed.
        self.pressure_dim = 30   # 30 pressure sensors in one layer on the rectangular cylinder

        # ---------- Condition encoder (MLP) ----------
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.pressure_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # ---------- Flow field encoder (CNN) ----------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.input_channels + 1, 32, kernel_size=4, stride=2, padding=1),  # (2+1) -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # compute flattened feature size (for FC)
        dummy_input = torch.zeros(1, 3, 367, 383)  # 2 flow + 1 condition channel
        dummy_output = self.encoder_conv(dummy_input)
        self.flattened_size = dummy_output.shape[1]

        # Latent mean and logvar
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # ---------- Decoder (latent + pressure) ----------
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + 128, 128 * 16),
            nn.ReLU(),
            nn.Linear(128 * 16, 128 * 48),
            nn.ReLU(),
            nn.Linear(128 * 48, 128 * 46 * 48)
            )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 92, 96)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 184, 192)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1),    # -> (2, 368, 384)
            nn.Tanh()  # assuming normalized velocity fields
        )

    def encode(self, x, p):
        B = x.shape[0]
        p_encoded = self.condition_encoder(p)  # (B, 128)
        p_map = p_encoded.view(B, 128, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
        #p_map = p_map.squeeze(1)  # -> (B, H, W)
        #p_map = p_map.unsqueeze(1)  # (B, 1, H, W)

        x_cat = torch.cat([x, p_map], dim=1)  # (B, 3, H, W)
        x_feat = self.encoder_conv(x_cat)
        mu = self.fc_mu(x_feat)
        logvar = self.fc_logvar(x_feat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, p):
        p_encoded = self.condition_encoder(p)  # (B, 128)
        z_cat = torch.cat([z, p_encoded], dim=1)
        x = self.decoder_input(z_cat)  # (B, 128 * 46 * 48)
        x = x.view(-1, 128, 46, 48)
        x = self.decoder_conv(x)
        x = x[:, :, :367, :383]  # crop to original shape if needed
        return x

    def forward(self, x, p):
        mu, logvar = self.encode(x, p)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, p)
        return x_recon, mu, logvar
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    """
    Conditional VAE for 2-channel flow fields with a pressure condition vector.

    Key fixes and behaviours:
    - Accepts x: (B, 2, H, W) and p: (B, pressure_dim).
    - Encodes pressure to a cond_dim vector and tiles it to a (B, cond_dim, H, W) map.
    - Dynamically computes the flattened encoder feature size using a dummy input so the
      linear layers always match the convolutional output for the provided input size.
    - Decoder projects (z + cond) back to the flattened conv feature size, reshapes to
      (B, 128, Hf, Wf) and uses ConvTranspose to upsample. The final output is
      resized (interpolated) to the original input (H, W) — robust to small mismatches.
    - Reparameterization returns mu during eval mode (deterministic inference).
    """

    def __init__(self,
                 latent_dim: int = 128,
                 pressure_dim: int = 30,
                 cond_dim: int = 128,
                 input_channels: int = 2,
                 input_height: int = 383,
                 input_width: int = 367):
        super().__init__()
        self.latent_dim = latent_dim
        self.pressure_dim = pressure_dim
        self.cond_dim = cond_dim
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        # ---------- Condition encoder (MLP for pressure) ----------
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.pressure_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.cond_dim),
            nn.ReLU()
        )

        # ---------- Flow field encoder (CNN) ----------
        enc_in_ch = self.input_channels + self.cond_dim # 2+128
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(enc_in_ch, 32, kernel_size=4, stride=2, padding=1),  # downsample x2  1, 32, 191, 183
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),   # 1, 64, 95, 91
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # 1, 128, 47, 45
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), # 1, 128, 23, 22
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), # 1, 128, 11, 11
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), # 1, 128, 5, 5
            nn.ReLU(),  # there will be a linear in the fc_mu and fc_logvar layer 
        )

        # compute flattened feature size (for FC) using a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, enc_in_ch, self.input_height, self.input_width)  # 1, 130, 383, 367
            out = self.encoder_conv(dummy)
            # out shape: (1, 128, 5, 5)
            self.hf = out.shape[2]  # 5
            self.wf = out.shape[3]  # 5
            self.flattened_size = out.shape[1] * out.shape[2] * out.shape[3] # 128 * 5 * 5

        # Latent mean and logvar
        """
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flattened_size, self.latent_dim * 16),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 16, 8 * self.latent_dim),
            nn.ReLU(),
            nn.Linear(8 * self.latent_dim, 2 * self.latent_dim),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim)
        )"""
        self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim)  # 128 * 5 * 5  --> 128
        self.fc_logvar = nn.Linear(self.flattened_size, self.latent_dim)  # 128 * 5 * 5  --> 128
        """
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.flattened_size, self.latent_dim * 16),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 16, 8 * self.latent_dim),
            nn.ReLU(),
            nn.Linear(8 * self.latent_dim, 2 * self.latent_dim),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim)
        )
        """
        # ---------- Decoder (latent + pressure) ----------
        # self.decoder_input = nn.Linear(self.latent_dim + self.cond_dim, self.flattened_size)  # 128 + 128 --> 128 * 5 * 5
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + 128, 128 * 16),
            nn.ReLU(),
            nn.Linear(128 * 16, 128 * 5*5),
            #nn.ReLU(),
            #nn.Linear(128 * 48, 128 * 46 * 48)
            )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor, p: torch.Tensor):
        """Encode x (B,2,H,W) conditioned on p (B,pressure_dim). Returns mu, logvar."""
        B = x.shape[0]
        p_encoded = self.condition_encoder(p)  # (B, cond_dim)
        # expand to spatial map: (B, cond_dim, H, W)
        p_map = p_encoded.view(B, self.cond_dim, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])

        x_cat = torch.cat([x, p_map], dim=1)  # (B, input_ch + cond_dim, H, W)
        features = self.encoder_conv(x_cat)  # (B, 128, Hf, Wf)
        features_flat = features.view(B, -1)

        mu = self.fc_mu(features_flat)
        logvar = self.fc_logvar(features_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # deterministic in eval mode
            return mu

    def decode(self, z: torch.Tensor, p: torch.Tensor):
        B = z.shape[0]
        p_encoded = self.condition_encoder(p)  # (B, cond_dim)
        z_cat = torch.cat([z, p_encoded], dim=1)  # (B, latent + cond_dim)

        x = self.decoder_input(z_cat)  # (B, flattened_size)
        x = x.view(B, 128, self.hf, self.wf)  # reshape to conv feature map  1, 128, 5, 5
        x = self.decoder_conv(x)  # upsampled, maybe not exactly input size

        # robustly resize to the original input size
        x = F.interpolate(x, size=(self.input_height, self.input_width), mode='bilinear', align_corners=False)
        return x

    def forward(self, x: torch.Tensor, p: torch.Tensor):
        mu, logvar = self.encode(x, p)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, p)
        return x_recon, mu, logvar
