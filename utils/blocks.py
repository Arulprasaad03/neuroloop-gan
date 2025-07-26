import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, downsample=False):
        super(ResidualBlock, self).__init__()
        self.upsample = upsample
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        identity = x
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            identity = F.interpolate(identity, scale_factor=2, mode='nearest')
        elif self.downsample:
            x = F.avg_pool2d(x, 2)
            identity = F.avg_pool2d(identity, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.skip_conv(identity)
        return F.relu(x + identity)

# Unused in unconditional GAN
class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, embed_size):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_size)
    def forward(self, labels):
        return self.embed(labels)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_label_tensor(label, num, device):
    return torch.full((num,), label, dtype=torch.long).to(device)