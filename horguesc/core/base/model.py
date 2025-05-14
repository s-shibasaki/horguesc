import abc
import torch.nn as nn

class BaseModel(nn.Module, abc.ABC):
    """Base model class that all task-specific models should inherit from."""
    
    def __init__(self, config, encoder=None):
        super().__init__()
        self.config = config
        self.encoder = encoder
    
    @abc.abstractmethod
    def forward(self, inputs):
        """Forward pass for the model.
        
        Args:
            inputs: Input data for the model
            
        Returns:
            dict: Dictionary containing model outputs
        """
        pass
    
    @abc.abstractmethod
    def compute_loss(self, outputs, targets):
        """Compute the loss for this model.
        
        Args:
            outputs: Output from the forward pass
            targets: Target values
            
        Returns:
            torch.Tensor: Loss value
        """
        pass
    
    @abc.abstractmethod
    def compute_metrics(self, outputs, targets):
        """Compute task-specific evaluation metrics.
        
        Args:
            outputs: Output from the forward pass
            targets: Target values
            
        Returns:
            dict: Dictionary containing metric names and values
        """
        pass
    
    @abc.abstractmethod
    def get_name(self):
        """Get the name of the model.
        
        Returns:
            str: Model name
        """
        pass
