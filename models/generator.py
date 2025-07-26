import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from utils.blocks import ResidualBlock

class Generator(nn.Module):
    def __init__(self, z_dim=100, base_filters=64, image_size=64):
        super().__init__()
        self.z_dim = z_dim
        self.init_size = 4
        self.image_size = image_size
        self.initial = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(z_dim, base_filters * 8, self.init_size, 1, 0)),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(True)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_filters * 8, base_filters * 4, upsample=True),  # 4x4 -> 8x8
            ResidualBlock(base_filters * 4, base_filters * 2, upsample=True),  # 8x8 -> 16x16
            ResidualBlock(base_filters * 2, base_filters, upsample=True),      # 16x16 -> 32x32
            ResidualBlock(base_filters, base_filters, upsample=True)           # 32x32 -> 64x64
        )
        self.output = nn.Sequential(
            spectral_norm(nn.Conv2d(base_filters, 3, 3, 1, 1)),
            nn.Tanh()
        )

    def forward(self, z, feedback=None):
        if feedback is not None:
            z = z + feedback
        out = self.initial(z.view(-1, self.z_dim, 1, 1))
        out = self.res_blocks(out)
        out = self.output(out)
        if out.shape[2:] != (self.image_size, self.image_size):
            raise ValueError(f"Generator output shape {out.shape[2:]} does not match expected ({self.image_size}, {self.image_size})")
        return out