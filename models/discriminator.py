import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.blocks import ResidualBlock
from configs import cfg
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.initial_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.res1 = ResidualBlock(64, 128, downsample=True)
        self.res2 = ResidualBlock(128, 256, downsample=True)
        self.res3 = ResidualBlock(256, 512, downsample=True)
        self.res4 = ResidualBlock(512, 1024, downsample=True)
        self.flatten = nn.Flatten()
        self.adv_out = spectral_norm(nn.Linear(1024 * 4 * 4, 1))  # 64x64 -> 4x4

    def forward(self, img, labels=None):
        out = self.initial_conv(img)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        flat = self.flatten(out)
        adv_score = self.adv_out(flat)
        return adv_score

    def extract_features(self, img):
        out = self.initial_conv(img)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        pooled = F.adaptive_avg_pool2d(out, (1, 1))  # shape: (batch_size, 1024, 1, 1)
        return pooled.view(pooled.size(0), -1)  # shape: (batch_size, 1024)