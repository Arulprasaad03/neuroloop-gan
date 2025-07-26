# loopgan_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.blocks import ResidualBlock
from configs import cfg

class ExpressionClassifier(nn.Module):
    def __init__(self):
        super(ExpressionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(cfg.channels, 64, 3, padding=1)
        self.res1 = ResidualBlock(64, 128, downsample=True)
        self.res2 = ResidualBlock(128, 256, downsample=True)
        self.res3 = ResidualBlock(256, 512, downsample=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, cfg.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
