"""
Implementation of the train command for horguesc.
"""
import logging
import sys
import torch
from importlib import import_module
from horguesc.utils.config import load_config
from horguesc.core.trainer import MultitaskTrainer

logger = logging.getLogger(__name__)

def run(args):
    """
    Run the train command.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code
    """
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize models and datasets for each task
        models = {}
        datasets = {}
        all_parameters = []
        
        # Get tasks from config
        tasks = config.get('training', 'tasks').split(',')
        
        for task in tasks:
            task = task.strip()
            try:
                # Import task modules
                model_module = import_module(f"horguesc.tasks.{task}.model")
                dataset_module = import_module(f"horguesc.tasks.{task}.dataset")
                
                # Initialize model and dataset
                model_class = getattr(model_module, f"{task.capitalize()}Model")
                dataset_class = getattr(dataset_module, f"{task.capitalize()}Dataset")
                
                model = model_class(config)
                dataset = dataset_class(config)
                
                models[task] = model
                datasets[task] = dataset
                
                # Collect parameters for optimizer
                all_parameters.extend(model.parameters())
                
                logger.info(f"Initialized model and dataset for task: {task}")
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to initialize task {task}: {e}")
                return 1
        
        # Create optimizer
        optimizer_name = config.get('training', 'optimizer', fallback='Adam')
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(
            all_parameters,
            lr=config.getfloat('training', 'learning_rate', fallback=0.001)
        )
        
        # Initialize trainer and train model
        trainer = MultitaskTrainer(config, models, datasets, optimizer)
        num_epochs = config.getint('training', 'num_epochs', fallback=10)
        trainer.train(num_epochs)
        
        logger.info("Training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1