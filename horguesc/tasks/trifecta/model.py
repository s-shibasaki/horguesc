import torch
import torch.nn as nn
import torch.nn.functional as F
from horguesc.core.base.model import BaseModel

class TrifectaModel(BaseModel):
    """Model for trifecta prediction task."""
    
    def __init__(self, config):
        super().__init__(config)
        # Parse model architecture from config
        input_dim = config.getint('models.trifecta', 'input_dim', fallback=10)
        hidden_dim = config.getint('models.trifecta', 'hidden_dim', fallback=64)
        output_dim = config.getint('models.trifecta', 'output_dim', fallback=3)
        
        # Define model layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(config.getfloat('models.trifecta', 'dropout', fallback=0.2))
    
    def forward(self, inputs):
        """Forward pass for trifecta model.
        
        Args:
            inputs: Dictionary containing input features
            
        Returns:
            dict: Dictionary containing model outputs
        """
        x = inputs['features']
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return {'logits': logits}
    
    def compute_loss(self, outputs, targets):
        """Compute loss for trifecta prediction.
        
        Args:
            outputs: Dictionary containing model outputs
            targets: Dictionary containing target values
            
        Returns:
            torch.Tensor: Loss value
        """
        logits = outputs['logits']
        labels = targets['labels']
        return F.cross_entropy(logits, labels)
    
    def get_name(self):
        """Get model name.
        
        Returns:
            str: Model name
        """
        return "trifecta"