import logging
import torch
from collections import defaultdict
import configparser

logger = logging.getLogger(__name__)

class MultitaskTrainer:
    """Trainer for multi-task learning."""
    
    def __init__(self, config, models, datasets, optimizer):
        """Initialize the multi-task trainer.
        
        Args:
            config: Configuration object
            models: Dictionary of task_name -> model
            datasets: Dictionary of task_name.mode -> dataset (e.g. "trifecta.train", "trifecta.eval")
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
            
        # Dictionary to store training and eval datasets by task
        self.task_datasets = {}
        for task_key, dataset in self.datasets.items():
            if "." in task_key:
                task_name, mode = task_key.split(".")
                if task_name not in self.task_datasets:
                    self.task_datasets[task_name] = {}
                self.task_datasets[task_name][mode] = dataset
    
    def train(self, num_epochs):
        """Train all models for the specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
        """
        logger.info(f"Starting multi-task training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            
            # Save models after each epoch if configured
            if self.config.getboolean('training', 'save_each_epoch', fallback=False):
                self._save_models(epoch)
    
    def _train_epoch(self, epoch):
        """Train all models for one epoch.
        
        Args:
            epoch: Current epoch number
        """
        total_loss = 0
        task_losses = defaultdict(float)
        
        # Get steps_per_epoch with proper error handling
        try:
            steps = self.config.getint('training', 'steps_per_epoch')
        except (ValueError, KeyError, configparser.NoOptionError, configparser.NoSectionError):
            # Better fallback logic
            steps = 100
            logger.warning(f"'steps_per_epoch' not properly configured, using default value: {steps}")
        
        # Set all models to training mode
        for model in self.models.values():
            model.train()
        
        for step in range(steps):
            # Zero out gradients
            self.optimizer.zero_grad()
            
            # Accumulate loss from all tasks
            combined_loss = 0
            
            for task_name, model in self.models.items():
                if task_name not in self.task_datasets or 'train' not in self.task_datasets[task_name]:
                    logger.warning(f"No training dataset found for task: {task_name}")
                    continue
                
                # Get the training dataset for this task
                dataset = self.task_datasets[task_name]['train']
                
                # Get batch for this task
                batch = dataset.get_batch()
                
                # Create inputs and targets dictionaries
                inputs = {}
                targets = {}
                
                # Separate inputs and targets based on their names
                for key, value in batch.items():
                    if key == 'target' or key.startswith('target_'):
                        targets[key] = value.detach().clone().to(self.device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=self.device)
                    else:
                        inputs[key] = value.detach().clone().to(self.device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=self.device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = model.compute_loss(outputs, targets)
                
                # Weight the loss (optional)
                try:
                    task_weight = self.config.getfloat(f'tasks.{task_name}', 'weight', fallback=1.0)
                except (ValueError, KeyError):
                    task_weight = 1.0
                    logger.debug(f"Could not find weight for task {task_name}, using default weight: 1.0")
                
                weighted_loss = loss * task_weight
                
                # Add to combined loss
                combined_loss += weighted_loss
                task_losses[task_name] += loss.item()
            
            # Backward pass
            combined_loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            total_loss += combined_loss.item()
            
            log_interval = self.config.getint('training', 'log_interval', fallback=10)
            if step % log_interval == 0:
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
            
            for task_name, model in self.models.items():
                if task_name not in self.task_datasets or 'eval' not in self.task_datasets[task_name]:
                    logger.warning(f"No validation dataset found for task: {task_name}")
                    continue
                
                # Get the validation dataset for this task
                dataset = self.task_datasets[task_name]['eval']
                
                # Get all validation data
                val_data = dataset.get_all_data()
                
                # Create inputs and targets dictionaries
                inputs = {}
                targets = {}
                
                # Separate inputs and targets
                for key, value in val_data.items():
                    if key == 'target' or key.startswith('target_'):
                        targets[key] = value.detach().clone().to(self.device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=self.device)
                    else:
                        inputs[key] = value.detach().clone().to(self.device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=self.device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = model.compute_loss(outputs, targets)
                task_metrics[task_name] = {'loss': loss.item()}
                
                # Additional metrics could be computed here
                self._compute_additional_metrics(task_name, outputs, targets, task_metrics)
            
            # Log validation results
            logger.info(f"Validation after epoch {epoch}:")
            for task_name, metrics in task_metrics.items():
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                logger.info(f"  {task_name}: {metrics_str}")
    
    def _compute_additional_metrics(self, task_name, outputs, targets, metrics_dict):
        """Compute additional metrics for validation."""
        # Get the model for this task
        model = self.models[task_name]
        
        # Call the model's compute_metrics method if it exists
        if hasattr(model, 'compute_metrics'):
            task_metrics = model.compute_metrics(outputs, targets)
            metrics_dict[task_name].update(task_metrics)
    
    def _save_models(self, epoch):
        """Save models after an epoch."""
        path_prefix = f"epoch_{epoch}"
        self.save_models(path_prefix)
        
    def save_models(self, path_prefix=None):
        """Save all models.
    
        Args:
            path_prefix: Optional path prefix for model files
        """
        try:
            import os
        
            # Get model directory from config or use default
            model_dir = self.config.get('paths', 'model_dir', fallback='models')
            os.makedirs(model_dir, exist_ok=True)
        
            # Use provided prefix or create a timestamp-based one
            if path_prefix is None:
                from datetime import datetime
                path_prefix = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
            # Save each model
            for task_name, model in self.models.items():
                model_path = os.path.join(model_dir, f"{path_prefix}_{task_name}.pt")
            
                # Save model state and metadata
                model_data = {
                    'state_dict': model.state_dict(),
                    'model_name': model.get_name(),
                    'task': task_name,
                    'config': {k: dict(self.config[k]) for k in self.config.sections()} 
                          if hasattr(self.config, 'sections') else {},
                }
            
                torch.save(model_data, model_path)
                logger.info(f"Saved model for task {task_name} to {model_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
        
    def load_models(self, path_prefix, device=None):
        """Load all models from saved files.
    
        Args:
            path_prefix: Path prefix for model files
            device: Optional device to load models to
    
        Returns:
            bool: True if all models were loaded successfully
        """
        try:
            import os
            import glob
        
            if device is None:
                device = self.device
        
            # Get model directory from config
            model_dir = self.config.get('paths', 'model_dir', fallback='models')
        
            # Find all model files with this prefix
            model_pattern = os.path.join(model_dir, f"{path_prefix}_*.pt")
            model_files = glob.glob(model_pattern)
        
            if not model_files:
                logger.warning(f"No model files found matching pattern: {model_pattern}")
                return False
        
            # Load each model
            for model_path in model_files:
                # Extract task name from filename
                filename = os.path.basename(model_path)
                task_name = filename[len(path_prefix)+1:-3]  # Remove prefix_ and .pt
            
                if task_name not in self.models:
                    logger.warning(f"Task {task_name} not found in current models. Skipping.")
                    continue
                
                # Load model data
                model_data = torch.load(model_path, map_location=device)
            
                # Load state dict into model
                self.models[task_name].load_state_dict(model_data['state_dict'])
                self.models[task_name].to(device)
            
                logger.info(f"Loaded model for task {task_name} from {model_path}")
        
            return True
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False