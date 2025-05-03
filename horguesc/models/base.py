"""
Base model class for all models.
"""
import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    """Base model class for all models."""

    def __init__(self, task_name="base"):
        """Initialize the base model."""
        super(BaseModel, self).__init__()
        self.task_name = task_name

    def save(self, path):
        """Save model to the specified path."""
        torch.save(self.state_dict(), path)
        logger.info(f"Model for task '{self.task_name}' saved to {path}.")

    def load(self, path):
        """Load model from the specified path."""
        self.load_state_dict(torch.load(path))
        logger.info(f"Model for task '{self.task_name}' loaded from {path}.")

        
       