"""
Base dataset class for all datasets.
"""
import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class BaseRacingDataset(Dataset):
    """Base class for all racing datasets."""
    
    def __init__(self, db_ops, config=None):
        """
        Initialize base dataset.
        
        Args:
            db_ops: DatabaseOperations instance
            config: Optional configuration dictionary
        """
        self.db_ops = db_ops
        self.config = config or {}
        self.data = []
        self.features = []
        self.labels = []
    
    def load_data(self):
        """Load data from database - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_data method")
    
    def preprocess_data(self):
        """Preprocess data - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement preprocess_data method")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]