import abc

class BaseDataset(abc.ABC):
    """Base dataset class that all task-specific datasets should inherit from."""
    
    def __init__(self, config):
        self.config = config
        
    @abc.abstractmethod
    def get_next_batch(self, batch_size):
        """Get next batch of data for training.
        
        Args:
            batch_size: Size of the batch to retrieve
            
        Returns:
            dict: Dictionary containing inputs and targets for the task
        """
        pass
    
    @abc.abstractmethod
    def get_validation_data(self):
        """Get validation data for evaluation.
        
        Returns:
            dict: Dictionary containing validation inputs and targets
        """
        pass
    
    @abc.abstractmethod
    def get_name(self):
        """Get the name of the task.
        
        Returns:
            str: Task name
        """
        pass