"""
Dataset export command for horguesc.

This module provides functionality to export dataset samples to Excel files
for inspection and validation of the data pipeline.
"""
import logging
import os
import sys
import pandas as pd
import numpy as np
from importlib import import_module
from datetime import datetime
from horguesc.core.features.processor import FeatureProcessor

logger = logging.getLogger(__name__)

def run(config):
    """
    Run the dataset export command.
    
    Args:
        config: Application configuration
        
    Returns:
        int: Exit code
    """
    try:
        # Get export parameters from config
        mode = config.get('export', 'mode', fallback='train')
        start_date = config.get('export', 'start_date', fallback=None)
        end_date = config.get('export', 'end_date', fallback=None)
        sample_size = config.getint('export', 'sample_size', fallback=100)
        output_dir = config.get('export', 'output_dir', fallback='exports')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get tasks from config
        tasks_str = config.get('tasks', 'active', fallback='')
        tasks = [task.strip() for task in tasks_str.split(',') if task.strip()]
        
        if not tasks:
            logger.error("No active tasks configured. Please set tasks.active in your config.")
            return 1
        
        logger.info(f"Exporting datasets for tasks: {', '.join(tasks)}")
        
        # Create feature processor
        feature_processor = FeatureProcessor(config)
        
        # Process each task
        for task in tasks:
            logger.info(f"Processing task: {task}")
            
            # Try to import the dataset module
            try:
                dataset_module = import_module(f"horguesc.tasks.{task}.dataset")
                dataset_class = getattr(dataset_module, f"{task.capitalize()}Dataset")
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to import dataset for task {task}: {e}")
                continue
            
            # Create the dataset
            try:
                dataset = dataset_class(
                    config=config,
                    mode=mode,
                    batch_size=sample_size,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Fetch raw data
                logger.info(f"Fetching data for {task}...")
                dataset.fetch_data()
                
                if not dataset.raw_data or not any(dataset.raw_data.values()):
                    logger.warning(f"No data found for task {task} with date range {start_date} to {end_date}")
                    continue
                
                # Export raw data
                export_dataset_to_excel(dataset.raw_data, task, "raw", output_dir)
                
                # Process features if in train or eval mode
                if mode in ['train', 'eval']:
                    # Collect feature values
                    dataset.collect_features(feature_processor)
                    feature_processor.fit()
                    
                    # Process features
                    dataset.process_features(feature_processor)
                    
                    # Export processed data
                    export_processed_data_to_excel(dataset.processed_data, task, output_dir)
                
                logger.info(f"Successfully exported data for task {task}")
                
            except Exception as e:
                logger.error(f"Error processing dataset for task {task}: {e}", exc_info=True)
                continue
        
        logger.info(f"All exports completed. Files saved to: {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error in export command: {e}", exc_info=True)
        return 1

def export_dataset_to_excel(data_dict, task_name, data_type, output_dir):
    """
    Export dataset data to Excel.
    
    Args:
        data_dict: Dictionary containing dataset data
        task_name: Name of the task
        data_type: Type of data ('raw' or 'processed')
        output_dir: Output directory
    """
    # Get timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"{task_name}_{data_type}_{timestamp}.xlsx")
    
    logger.info(f"Exporting {data_type} data for task {task_name} to {output_file}")
    
    # Create Excel writer
    with pd.ExcelWriter(output_file) as writer:
        # First sheet: General information about the data
        keys = list(data_dict.keys())
        info_df = pd.DataFrame({
            'Key': keys,
            'Type': [type(data_dict[k]).__name__ for k in keys],
            'Shape': [getattr(data_dict[k], 'shape', 'N/A') if hasattr(data_dict[k], 'shape') else len(data_dict[k]) if hasattr(data_dict[k], '__len__') else 'N/A' for k in keys]
        })
        info_df.to_excel(writer, sheet_name='DataInfo', index=False)
        
        # Identify race IDs if available
        race_ids = data_dict.get('kyoso_id', None)
        
        # For each data type, create an appropriate representation
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    # Convert 1D array to DataFrame
                    df = pd.DataFrame({key: value})
                    df.to_excel(writer, sheet_name=key[:31], index=True)  # Limit sheet name to 31 chars
                elif value.ndim == 2:
                    # For 2D arrays, create a DataFrame with appropriate headers
                    if key == 'horse_count':
                        # Special handling for horse_count
                        df = pd.DataFrame({key: value})
                    else:
                        # Create column names for each horse
                        columns = [f"Horse_{i+1}" for i in range(value.shape[1])]
                        df = pd.DataFrame(value, columns=columns)
                        
                        # Add race ID as index if available
                        if race_ids is not None and len(race_ids) == value.shape[0]:
                            df.index = race_ids
                            df.index.name = 'race_id'
                    
                    df.to_excel(writer, sheet_name=key[:31])
                else:
                    # For higher dimension arrays, just save shape information
                    logger.info(f"Skipping {key} with shape {value.shape} (dimension > 2)")
            elif isinstance(value, list) and value and all(isinstance(x, (str, int, float, bool, type(None))) for x in value):
                # For simple lists, convert to DataFrame
                df = pd.DataFrame({key: value})
                df.to_excel(writer, sheet_name=key[:31], index=True)
            else:
                logger.info(f"Skipping {key} with type {type(value).__name__}")

def export_processed_data_to_excel(data_dict, task_name, output_dir):
    """
    Export processed dataset data to Excel.
    
    Args:
        data_dict: Dictionary containing processed dataset data
        task_name: Name of the task
        output_dir: Output directory
    """
    # Get timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"{task_name}_processed_{timestamp}.xlsx")
    
    logger.info(f"Exporting processed data for task {task_name} to {output_file}")
    
    # Create Excel writer
    with pd.ExcelWriter(output_file) as writer:
        # First sheet: General information about the data
        keys = list(data_dict.keys())
        info_df = pd.DataFrame({
            'Key': keys,
            'Type': [type(data_dict[k]).__name__ for k in keys],
            'Shape': [getattr(data_dict[k], 'shape', 'N/A') if isinstance(data_dict[k], (np.ndarray, pd.DataFrame)) or 
                     hasattr(data_dict[k], 'shape') else len(data_dict[k]) if hasattr(data_dict[k], '__len__') else 'N/A' for k in keys]
        })
        info_df.to_excel(writer, sheet_name='ProcessedDataInfo', index=False)
        
        # Identify race IDs if available
        race_ids = None
        if 'kyoso_id' in data_dict:
            race_ids = data_dict['kyoso_id']
        
        # For each data type, create an appropriate representation
        for key, value in data_dict.items():
            try:
                if isinstance(value, np.ndarray):
                    # Handle numpy arrays
                    export_numpy_data(key, value, writer, race_ids)
                elif hasattr(value, 'numpy') and callable(getattr(value, 'numpy')):
                    # Handle torch tensors
                    numpy_data = value.detach().cpu().numpy()
                    export_numpy_data(key, numpy_data, writer, race_ids)
                elif isinstance(value, list) and value and all(isinstance(x, (str, int, float, bool, type(None))) for x in value):
                    # For simple lists, convert to DataFrame
                    df = pd.DataFrame({key: value})
                    df.to_excel(writer, sheet_name=key[:31], index=True)
                else:
                    logger.info(f"Skipping {key} with type {type(value).__name__}")
            except Exception as e:
                logger.error(f"Error exporting {key}: {e}")

def export_numpy_data(key, value, writer, race_ids=None):
    """Helper function to export numpy data to Excel."""
    if value.ndim == 1:
        # Convert 1D array to DataFrame
        df = pd.DataFrame({key: value})
        df.to_excel(writer, sheet_name=key[:31], index=True)  # Limit sheet name to 31 chars
    elif value.ndim == 2:
        # For 2D arrays, create a DataFrame with appropriate headers
        if key == 'horse_count':
            # Special handling for horse_count
            df = pd.DataFrame({key: value})
        else:
            # Create column names for each horse/feature
            if value.shape[1] > 100:  # Too many columns to show all
                # Sample a subset of columns
                sampled_cols = np.random.choice(value.shape[1], size=min(100, value.shape[1]), replace=False)
                sampled_vals = value[:, sampled_cols]
                columns = [f"Feature_{i}" for i in sampled_cols]
                df = pd.DataFrame(sampled_vals, columns=columns)
                logger.info(f"Sampled {len(columns)} out of {value.shape[1]} columns for {key}")
            else:
                columns = [f"Feature_{i+1}" for i in range(value.shape[1])]
                df = pd.DataFrame(value, columns=columns)
            
            # Add race ID as index if available
            if race_ids is not None and len(race_ids) == value.shape[0]:
                df.index = race_ids
                df.index.name = 'race_id'
        
        df.to_excel(writer, sheet_name=key[:31])
    else:
        # For higher dimension arrays, export shape information
        sample_df = pd.DataFrame({
            'Shape': [str(value.shape)],
            'Dimension': [value.ndim],
            'DataType': [str(value.dtype)]
        })
        sample_df.to_excel(writer, sheet_name=f"{key[:27]}_info")