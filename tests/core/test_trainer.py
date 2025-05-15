import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from horguesc.core.trainer import MultitaskTrainer


class SimpleModel(torch.nn.Module):
    """Simple model for testing"""
    
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        
    def forward(self, inputs):
        """Simple forward pass"""
        x = inputs['feature']
        return {'output': self.linear(x)}
    
    def compute_loss(self, outputs, targets):
        """Simple loss calculation"""
        return torch.nn.functional.mse_loss(outputs['output'], targets['target'])
    
    def compute_metrics(self, outputs, targets):
        """Simple metrics calculation"""
        with torch.no_grad():
            mse = torch.mean((outputs['output'] - targets['target'])**2).item()
        return {'mse': mse}
    
    def get_name(self):
        """Return model name"""
        return "SimpleModel"


class SimpleDataset:
    """Simple dataset for testing"""
    
    def __init__(self, n_samples=100, batch_size=10):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.data = {
            'feature': torch.randn(n_samples, 10),
            'target': torch.randn(n_samples, 1)
        }
        
    def get_batch(self):
        """Return a random batch of data"""
        indices = torch.randint(0, self.n_samples, (self.batch_size,))
        return {
            'feature': self.data['feature'][indices],
            'target': self.data['target'][indices]
        }
        
    def get_all_data(self):
        """Return all data"""
        return self.data


class TestMultitaskTrainer:
    """Tests for MultitaskTrainer"""
    
    @pytest.fixture
    def setup_trainer(self):
        """Fixture to set up a trainer instance with mocked components"""
        # Create a mock config
        config = MagicMock()
        config.getboolean.return_value = False
        config.getint.return_value = 5
        config.getfloat.return_value = 1.0
        
        # Create models dict
        models = {
            'task1': SimpleModel(),
            'task2': SimpleModel()
        }
        
        # Create datasets dict
        datasets = {
            'task1.train': SimpleDataset(n_samples=100, batch_size=10),
            'task1.eval': SimpleDataset(n_samples=50, batch_size=50),
            'task2.train': SimpleDataset(n_samples=100, batch_size=10),
            'task2.eval': SimpleDataset(n_samples=50, batch_size=50)
        }
        
        # Create optimizer
        optimizer = torch.optim.SGD(
            list(models['task1'].parameters()) + list(models['task2'].parameters()),
            lr=0.01
        )
        
        # Create and return trainer
        trainer = MultitaskTrainer(config, models, datasets, optimizer)
        return trainer, config, models, datasets, optimizer
    
    def test_initialization(self, setup_trainer):
        """Test initialization of MultitaskTrainer"""
        trainer, config, models, datasets, optimizer = setup_trainer
        
        # Check basic properties
        assert trainer.config == config
        assert trainer.models == models
        assert trainer.datasets == datasets
        assert trainer.optimizer == optimizer
        
        # Check if task datasets were properly organized
        assert 'task1' in trainer.task_datasets
        assert 'task2' in trainer.task_datasets
        assert 'train' in trainer.task_datasets['task1']
        assert 'eval' in trainer.task_datasets['task1']
        assert trainer.task_datasets['task1']['train'] == datasets['task1.train']
        assert trainer.task_datasets['task1']['eval'] == datasets['task1.eval']
    
    def test_train_epoch(self, setup_trainer):
        """Test _train_epoch method"""
        trainer, config, models, datasets, optimizer = setup_trainer
        
        # Mock the optimizer step to avoid actual parameter updates
        original_step = optimizer.step
        optimizer.step = MagicMock()
        
        # Run a training epoch
        trainer._train_epoch(0)
        
        # Check if optimizer.step was called
        assert optimizer.step.called
        
        # Restore original step method
        optimizer.step = original_step
    
    def test_validate_epoch(self, setup_trainer):
        """Test _validate_epoch method"""
        trainer, config, models, datasets, optimizer = setup_trainer
        
        # Run validation
        trainer._validate_epoch(0)
        
        # Nothing to assert directly, just ensuring it runs without errors
        # In a more comprehensive test, you might want to check metrics
    
    @patch('torch.save')
    @patch('os.makedirs')
    def test_save_models(self, mock_makedirs, mock_save, setup_trainer):
        """Test save_models method"""
        trainer, config, models, datasets, optimizer = setup_trainer
        
        # Configure mock config to return model_dir
        config.get.return_value = 'mock_model_dir'
        
        # Call save_models
        result = trainer.save_models(path_prefix="test_prefix")
        
        # Check if directories were created
        assert mock_makedirs.called
        
        # Check if torch.save was called for each model
        assert mock_save.call_count == len(models)
        
        # Check if the method returned success
        assert result is True
    
    def test_train(self, setup_trainer):
        """Test the full training loop"""
        trainer, config, models, datasets, optimizer = setup_trainer
        
        # Mock the internal methods
        trainer._train_epoch = MagicMock()
        trainer._validate_epoch = MagicMock()
        trainer._save_models = MagicMock()
        
        # Call train method
        num_epochs = 3
        trainer.train(num_epochs)
        
        # Check if the methods were called correctly
        assert trainer._train_epoch.call_count == num_epochs
        assert trainer._validate_epoch.call_count == num_epochs
        
        # _save_models should not be called since config.getboolean returns False
        assert trainer._save_models.call_count == 0


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])