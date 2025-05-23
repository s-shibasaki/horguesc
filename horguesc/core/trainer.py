import logging
import torch
from collections import defaultdict
import configparser
import numpy as np
import os
from datetime import datetime

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
        
        # For tracking best model metrics
        self.best_metrics = {}
        self.best_epoch = -1
        
    def train(self, num_epochs):
        """Train all models for the specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            dict: Final metrics from the last validation epoch if validation was performed, otherwise empty dict
        """
        logger.info(f"Starting multi-task training for {num_epochs} epochs")
        
        final_metrics = {}
        
        # Check if we should save models after each epoch
        save_each_epoch = self.config.getboolean('training', 'save_each_epoch', fallback=False)
        
        # Check if validation should be skipped
        skip_validation = self.config.getboolean('training', 'skip_validation', fallback=False)
        if skip_validation:
            logger.info("Validation will be skipped as requested")
        
        # Generate base path prefix for model saving
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path_prefix = f"model_{timestamp}"
        
        for epoch in range(num_epochs):
            self._train_epoch(epoch)
            
            # Only perform validation if not skipped
            if not skip_validation:
                metrics = self._validate_epoch(epoch)
                final_metrics = metrics  # 最後のエポックの評価指標を保存
            
            # Save model after each epoch if configured to do so
            if save_each_epoch:
                # Create epoch-specific path prefix
                epoch_path_prefix = f"{base_path_prefix}_epoch{epoch+1}"
                
                logger.info(f"Saving models for epoch {epoch+1}...")
                self.save_models(epoch_path_prefix)
                logger.info(f"Models for epoch {epoch+1} saved successfully")
    
        # After all epochs, save the final model if not already saved
        if not save_each_epoch:
            final_path_prefix = f"{base_path_prefix}_final"
            logger.info("Saving final models...")
            self.save_models(final_path_prefix)
            logger.info("Final models saved successfully")
            
        return final_metrics
    
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
                batch, _ = dataset.get_batch()  # Unpack tuple, ignore is_last_batch boolean
                
                # Create inputs and targets dictionaries
                inputs = {}
                
                # Separate inputs and targets based on their names
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.to(self.device)
                    else:
                        # For non-tensor inputs, just pass them as-is
                        inputs[key] = value
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = model.compute_loss(outputs, inputs)
                
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
                logger.info(f"Epoch {epoch+1}, Step {step}/{steps}, Loss: {combined_loss.item():.4f}")
    
        # Log epoch results
        avg_loss = total_loss / steps
        logger.info(f"Epoch {epoch+1} completed, Avg Loss: {avg_loss:.4f}")
        for task_name, loss in task_losses.items():
            logger.info(f"  {task_name} loss: {loss/steps:.4f}")
    
    def _validate_epoch(self, epoch):
        """Validate all models after an epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            dict: Validation metrics
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
                
                # Process all validation data using batches
                all_outputs = {}
                all_inputs = {}
                
                # Continue fetching batches until we've processed all validation data
                while True:
                    batch_data, is_last_batch = dataset.get_batch()
                    
                    # Create inputs and targets dictionaries
                    batch_inputs = {}
                    
                    # Separate inputs and targets
                    for key, value in batch_data.items():
                        if isinstance(value, torch.Tensor):
                            batch_inputs[key] = value.to(self.device)
                        else:
                            # For non-tensor inputs, just pass them as-is
                            batch_inputs[key] = value
                    
                    # Forward pass
                    batch_outputs = model(batch_inputs)
                    
                    # Accumulate outputs and targets
                    for key, value in batch_outputs.items():
                        if key not in all_outputs:
                            all_outputs[key] = []
                        all_outputs[key].append(value)
                    
                    for key, value in batch_inputs.items():
                        if key not in all_inputs:
                            all_inputs[key] = []
                        all_inputs[key].append(value)
                    
                    # If this was the last batch, break the loop
                    if is_last_batch:
                        break
                
                # Combine accumulated values based on their types
                combined_outputs = {}
                for key, values in all_outputs.items():
                    if not values:  # Skip empty lists
                        combined_outputs[key] = values
                        continue
                        
                    if isinstance(values[0], torch.Tensor):
                        # For PyTorch tensors, use torch.cat
                        combined_outputs[key] = torch.cat(values, dim=0)
                    elif isinstance(values[0], np.ndarray):
                        # For NumPy arrays, use np.concatenate
                        combined_outputs[key] = np.concatenate(values, axis=0)
                    elif isinstance(values[0], list):
                        # For lists, concatenate them
                        combined_list = []
                        for sublist in values:
                            combined_list.extend(sublist)
                        combined_outputs[key] = combined_list
                    else:
                        # For other types, keep the list of values
                        combined_outputs[key] = values
                
                combined_inputs = {}
                for key, values in all_inputs.items():
                    if not values:  # Skip empty lists
                        combined_inputs[key] = values
                        continue
                        
                    if isinstance(values[0], torch.Tensor):
                        # For PyTorch tensors, use torch.cat
                        combined_inputs[key] = torch.cat(values, dim=0)
                    elif isinstance(values[0], np.ndarray):
                        # For NumPy arrays, use np.concatenate
                        combined_inputs[key] = np.concatenate(values, axis=0)
                    elif isinstance(values[0], list):
                        # For lists, concatenate them
                        combined_list = []
                        for sublist in values:
                            combined_list.extend(sublist)
                        combined_inputs[key] = combined_list
                    else:
                        # For other types, keep the list of values
                        combined_inputs[key] = values
                
                # Compute loss
                loss = model.compute_loss(combined_outputs, combined_inputs)
                task_metrics[task_name] = {'loss': loss.item()}
                
                # Additional metrics could be computed here - Pass the epoch number
                self._compute_additional_metrics(task_name, combined_outputs, combined_inputs, task_metrics, epoch)
    
        # Log validation results
        logger.info(f"Validation after epoch {epoch+1}:")
        for task_name, metrics in task_metrics.items():
            # Flatten nested metrics with dot notation
            flat_metrics = self._flatten_metrics_dict(metrics)
            
            # Format the metrics string
            metrics_parts = []
            for k, v in flat_metrics.items():
                if isinstance(v, (int, float)):
                    # Format numbers with 4 decimal places
                    metrics_parts.append(f"{k}: {v:.4f}")
                else:
                    # Handle other types
                    metrics_parts.append(f"{k}: {v}")
            
            metrics_str = ", ".join(metrics_parts)
            logger.info(f"  {task_name}: {metrics_str}")

        return task_metrics

    def _flatten_metrics_dict(self, metrics_dict, parent_key=''):
        """Flatten a nested metrics dictionary using dot notation.
        
        Args:
            metrics_dict: The nested metrics dictionary
            parent_key: The parent key prefix (used for recursion)
            
        Returns:
            dict: Flattened metrics dictionary
        """
        flat_metrics = {}
        
        for k, v in metrics_dict.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            
            if isinstance(v, dict):
                # Recursively flatten nested dictionaries
                nested_flat = self._flatten_metrics_dict(v, new_key)
                flat_metrics.update(nested_flat)
            else:
                # Add the leaf node
                flat_metrics[new_key] = v
        
        return flat_metrics
    
    def _compute_additional_metrics(self, task_name, outputs, inputs, metrics_dict, epoch):
        """Compute additional metrics for validation.
        
        Args:
            task_name: Name of the task
            outputs: Model outputs
            inputs: Model inputs
            metrics_dict: Dictionary to store computed metrics
            epoch: Current epoch number
        """
        # Get the model for this task
        model = self.models[task_name]
        
        # Call the model's compute_metrics method if it exists
        if hasattr(model, 'compute_metrics'):
            task_metrics = model.compute_metrics(outputs, inputs, epoch)
            metrics_dict[task_name].update(task_metrics)
    
    def save_models(self, path_prefix=None):
        """Save all models.
    
        Args:
            path_prefix: Optional path prefix for model files
            
        Returns:
            bool: True if all models were saved successfully
        """
        try:
            # Get model directory from config or use default
            model_dir = self.config.get('paths', 'model_dir', fallback='models/default')
            os.makedirs(model_dir, exist_ok=True)
        
            # Use provided prefix or create a timestamp-based one
            if path_prefix is None:
                path_prefix = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
            # Save each model
            for task_name, model in self.models.items():
                model_path = os.path.join(model_dir, f"{path_prefix}_{task_name}.pt")
            
                # Save model state and metadata
                model_data = {
                    'state_dict': model.state_dict(),
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
            import glob
        
            if device is None:
                device = self.device
        
            # Get model directory from config
            model_dir = self.config.get('paths', 'model_dir', fallback='models/default')
        
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