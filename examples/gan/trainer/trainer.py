"""
Trainer workflow implementation that follows the framework's Trainer interface.
"""
import logging
import torch
import random
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Any, Optional
from framework.trainer.trainer import Trainer, TrainingStatus
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def remove_random_elements(arr, n):
  """Removes n random elements from a list.

  Args:
    arr: The list to remove elements from.
    n: The number of random elements to remove.

  Returns:
    A new list with n random elements removed.
  """
  if n >= len(arr):
    return []  # Or return an empty list if you prefer

  indices_to_remove = random.sample(range(len(arr)), n)
  new_arr = [item for i, item in enumerate(arr) if i not in indices_to_remove]
  return new_arr

class DataSet(Dataset):
    """
    DataSet is a class that contains the data for the training workflow.
    """
    def __init__(self, data: Any):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx].reshape(-1, 1)
        transformer = MinMaxScaler().fit(X[:self.observation_size])
        transformed = transformer.transform(X)
        return transformed[:self.seq_len].reshape(1, 1, -1), 1.0


class DataSets:
    """
    DataSets is a class that contains the data sets for the training workflow.
    """
    def __init__(self, min_samples: int, max_samples: int, split_ratio: float, sequence_length: int): 
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.split_ratio = split_ratio
        self.new_data_buffer = None
        self.training_set = []
        self.validation_set = []
        self.sequence_length = sequence_length

    def ingest_data(self, data: Any) -> None:
        """
        Ingests data from a data source to the training set.
        """
        if self.new_data_buffer is None:
            self.new_data_buffer = data
        else:
            self.new_data_buffer = pd.concat([self.new_data_buffer, data], ignore_index=True)

        if len(self.new_data_buffer) < self.sequence_length:
            return

        df = self.new_data_buffer[:self.sequence_length*int(len(self.new_data_buffer)/self.sequence_length)]
        self.new_data_buffer = self.new_data_buffer[self.sequence_length*int(len(self.new_data_buffer)/self.sequence_length):]
        # Process the new data buffer
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df = df.set_index('Timestamp')
        df = df.dropna()
        df = df.drop(columns=['Volume', 'Open', 'High', 'Low'])
        samples = np.array_split(df['Close'].values, len(df)/self.sequence_length)
        for sample in samples:
            if random.random() < self.split_ratio:
                self.validation_set.append(sample)
            else:
                self.training_set.append(sample)

        train_samples_to_remove = len(self.training_set) - int((self.max_samples * (1 - self.split_ratio)))
        if train_samples_to_remove > 0:
            self.training_set = remove_random_elements(self.training_set, train_samples_to_remove)

        validation_samples_to_remove = len(self.validation_set) - int(self.max_samples*self.split_ratio)
        if validation_samples_to_remove > 0:
            self.validation_set = remove_random_elements(self.validation_set, validation_samples_to_remove)

        print(f"Training set size: {len(self.training_set)}")
        print(f"Validation set size: {len(self.validation_set)}")

    def get_training_set_size(self) -> int:
        """
        Returns the size of the training set.
        """
        return len(self.training_set)
    
    def get_training_set(self) -> Any:
        """
        Returns the training set.
        """
        return DataSet(self.training_set)
    
    def get_validation_set(self) -> Any:
        """
        Returns the validation set.
        """
        return DataSet(self.validation_set)
    

class LstmTrainerWithGanAugmentation(Trainer):
    """
    Trainer runs a training workflow that:
    1. Ingests data from a data source to the training set.
    2. Trains GAN model
    3. Trains Forecasting model on augmented data.
    """
    
    def __init__(self, min_samples: int):
        """
        Initialize the trainer.
        """
        self.data_sets = DataSets(min_samples=1000, max_samples=2000, split_ratio=0.2, sequence_length=90)
        self.training_status = TrainingStatus.GATHERING_DATA
        logger.info(f"Initialized GAN Trainer")


    def train(self, input_data: Any) -> None:
        """
        Run training workflow.
        """
        pass


    def get_status(self) -> TrainingStatus:
        """
        Get the current training status.
        
        Returns:
            TrainingStatus: Current status of the training process
        """
        return self.training_status


    def new_data(self, data: Any) -> None:
        """
        Ingest new data into the training process.
        """
        self.data_sets.ingest_data(data)
        if self.data_sets.get_training_set_size() >= self.data_sets.min_samples:
            self.training_status = TrainingStatus.DONE_GATHERING_DATA
