import os
import time
import torch
import random
import numpy as np
from torch import nn, optim
from torch.nn.utils import spectral_norm
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.autograd import Variable, grad
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import warnings
warnings.filterwarnings('ignore')

class Config:
    def __init__(self):
        # Dataset and training setup
        self.image_size = 64    # Set to 64x64 for targeted resolution
        self.batch_size = 128   # Increased to leverage more data per update
        self.epochs = 100       # Moderate; extend if needed for refinement
        self.num_workers = 4    # Sufficient for 64x64 data loading
        self.pin_memory = True  # Faster data transfer

        # Model architecture
        self.z_dim = 128        # Sufficient for 64x64 details

        # Learning rates
        self.lr_G = 0.00008     # Slightly increased for faster convergence
        self.lr_D = 0.000008    # Balanced with G for stability
        self.beta1 = 0.5        # Standard for Adam
        self.beta2 = 0.999      # Standard for Adam

        # WGAN-GP parameters
        self.n_critic = 5       # Balances D and G updates
        self.lambda_gp = 8.0    # Reduced for finer gradient control

        # Training enhancements
        self.noise_scale_init = 0.003  # Reduced for less disturbance
        self.noise_scale_min = 0.0003  # Reduced for cleaner outputs
        self.diversity_weight = 0.25   # Increased for better background variety
        self.save_freq = 5     # Adjust based on monitoring needs
        self.print_freq = 50   # Logging frequency
        self.result_dir = 'results_64x64_refined'  # For new runs
        self.model_dir = 'models_64x64_refined'    # For new checkpoints
        self.tensorboard_dir = 'runs_64x64_refined' # For new logs
        self.use_feedback_fusion = False           # Disable if unused

# Instantiate the config for global use
cfg = Config()