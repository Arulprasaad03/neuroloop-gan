import torch
import torch.nn as nn
from configs import cfg

class FeedbackEngine(nn.Module):
    def __init__(self, classifier=None, use_feedback=True):
        """
        Feedback engine for GAN training, using discriminator features when classifier is None.
        
        Args:
            classifier (nn.Module, optional): Classifier for feedback (unused if None).
            use_feedback (bool): Whether to apply feedback mechanism.
        """
        super(FeedbackEngine, self).__init__()
        self.use_feedback = use_feedback
        self.classifier = classifier
        # No projection layer needed; feedback_adapter in train_loopgan.py handles projection

    def forward(self, features):
        """
        Process features to generate feedback.
        
        Args:
            features (torch.Tensor): Features from discriminator.
            
        Returns:
            torch.Tensor: Processed feedback or None if not using feedback.
        """
        if not self.use_feedback:
            return None
        # Feedback is processed externally via feedback_adapter in train_loopgan.py
        return features