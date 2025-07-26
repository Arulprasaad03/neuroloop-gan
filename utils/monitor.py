import torch
import torch.nn as nn
from torchvision.models import resnet18

class ModeCollapseMonitor:
    def __init__(self, device, threshold=0.9):
        """
        Initialize the ModeCollapseMonitor with a ResNet-18 feature extractor.
        
        Args:
            device (torch.device): Device to run the feature extractor (e.g., 'cuda' or 'cpu').
            threshold (float): Similarity threshold for detecting mode collapse (default: 0.9).
        """
        self.device = device
        self.threshold = threshold
        self.prev_fakes = []
        self.current_similarity = 0.0  # Store the latest similarity score
        
        # Initialize ResNet-18 as feature extractor
        self.feature_extractor = resnet18(pretrained=True).to(device)
        self.feature_extractor.fc = nn.Identity()  # Remove final fully connected layer
        self.feature_extractor.eval()  # Set to evaluation mode
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Disable gradients to save memory

    def update_and_check(self, current_batch, threshold=None):
        """
        Update the monitor with the current batch and check for mode collapse.
        
        Args:
            current_batch (torch.Tensor): Batch of images (shape: [batch_size, 3, cfg.image_size, cfg.image_size]).
            threshold (float, optional): Override default threshold for this check.
        
        Returns:
            tuple: (collapsed, msg, similarity)
                - collapsed (bool): True if mode collapse detected (max similarity > threshold).
                - msg (str): Message describing the result (e.g., max similarity value).
                - similarity (float): The maximum similarity score from this batch.
        """
        threshold = threshold if threshold is not None else self.threshold
        
        # Cast input to FP32 for ResNet-18
        current_batch = current_batch.float()
        
        # Extract features using ResNet-18
        with torch.no_grad():
            features = self.feature_extractor(current_batch)  # Shape: [batch_size, 512]
            flat = features / (features.norm(dim=1, keepdim=True) + 1e-8)  # Normalize
        
        # Initialize previous fakes if empty
        if len(self.prev_fakes) == 0:
            self.prev_fakes = flat.detach()
            self.current_similarity = 0.0
            return False, "Initialized monitor.", self.current_similarity

        # Compute cosine similarity with previous batch
        sim = torch.mm(flat, self.prev_fakes.t())  # Shape: [batch_size, batch_size_prev]
        max_sim = sim.max().item()
        self.current_similarity = max_sim  # Update the latest similarity score

        # Update previous fakes
        self.prev_fakes = flat.detach()

        # Check for mode collapse
        if max_sim > threshold:
            return True, f"⚠️ Mode collapse detected! Max similarity: {max_sim:.2f}", max_sim
        return False, f"✅ No collapse. Max similarity: {max_sim:.2f}", max_sim