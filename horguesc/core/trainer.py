import logging
import torch
from collections import defaultdict

# TODO: Review entire file as related code has been modified. Need to ensure compatibility 
# with updated components and write comprehensive test cases for the MultitaskTrainer class.

logger = logging.getLogger(__name__)

class MultitaskTrainer:
    """Trainer for multi-task learning."""
    
    def __init__(self, config, models, datasets, optimizer):
        """Initialize the multi-task trainer.
        
        Args:
            config: Configuration object
            models: Dictionary of task_name -> model
            datasets: Dictionary of task_name -> dataset
            optimizer: Optimizer for all models
        """
        self.config = config
        self.models = models
        self.datasets = datasets
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        for model in self.models.values():
            model.to(self.device)
    
    def train(self, num_epochs):
        """Train all models for the specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
        """
        logger.info(f"Starting multi-task training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
    
    def _train_epoch(self, epoch):
        """Train all models for one epoch.
        
        Args:
            epoch: Current epoch number
        """
        total_loss = 0
        task_losses = defaultdict(float)
        steps = self.config.getint('training', 'steps_per_epoch')
        
        # Set all models to training mode
        for model in self.models.values():
            model.train()
        
        for step in range(steps):
            # Zero out gradients
            self.optimizer.zero_grad()
            
            # Accumulate loss from all tasks
            combined_loss = 0
            
            for task_name, dataset in self.datasets.items():
                # Get batch for this task
                batch = dataset.get_next_batch(self.config.getint('training', 'batch_size'))
                
                # Move batch to device
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch['inputs'].items()}
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch['targets'].items()}
                
                # Forward pass
                model = self.models[task_name]
                outputs = model(inputs)
                
                # Compute loss
                loss = model.compute_loss(outputs, targets)
                
                # Weight the loss (optional)
                task_weight = self.config.getfloat(f'tasks.{task_name}', 'weight', fallback=1.0)
                weighted_loss = loss * task_weight
                
                # Add to combined loss
                combined_loss += weighted_loss
                task_losses[task_name] += loss.item()
            
            # Backward pass
            combined_loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            total_loss += combined_loss.item()
            
            if step % self.config.getint('training', 'log_interval', fallback=10) == 0:
                logger.info(f"Epoch {epoch}, Step {step}/{steps}, Loss: {combined_loss.item():.4f}")
        
        # Log epoch results
        avg_loss = total_loss / steps
        logger.info(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")
        for task_name, loss in task_losses.items():
            logger.info(f"  {task_name} loss: {loss/steps:.4f}")
    
    def _validate_epoch(self, epoch):
        """Validate all models after an epoch.
        
        Args:
            epoch: Current epoch number
        """
        # Set all models to evaluation mode
        for model in self.models.values():
            model.eval()
        
        with torch.no_grad():
            task_metrics = {}
            
            for task_name, dataset in self.datasets.items():
                # Get validation data
                val_data = dataset.get_validation_data()
                
                # Move data to device
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in val_data['inputs'].items()}
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in val_data['targets'].items()}
                
                # Forward pass
                model = self.models[task_name]
                outputs = model(inputs)
                
                # Compute loss
                loss = model.compute_loss(outputs, targets)
                task_metrics[task_name] = {'loss': loss.item()}
                
                # Additional metrics could be computed here
            
            # Log validation results
            logger.info(f"Validation after epoch {epoch}:")
            for task_name, metrics in task_metrics.items():
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                logger.info(f"  {task_name}: {metrics_str}")