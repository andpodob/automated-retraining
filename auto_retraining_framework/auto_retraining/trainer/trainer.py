"""
Base class for implementing model training strategies.
"""

import threading
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum


class TrainingStatus(Enum):
    """Enumeration of possible training status codes."""
    TRAINING_IN_PROGRESS = "TRAINING_IN_PROGRESS"
    TRAINING_COMPLETED = "TRAINING_COMPLETED"
    TRAINING_FAILED = "TRAINING_FAILED"
    GATHERING_DATA = "GATHERING_DATA"
    READY = "READY"


class Trainer(ABC):
    """
    Base class for implementing model training strategies.
    Concrete implementations should inherit from this class and implement the train method.
    """
    
    def __init__(self):
        self._status_lock = threading.Lock()
        self.status = TrainingStatus.GATHERING_DATA

    @abstractmethod
    def train(self) -> None:
        """
        Train the model. After training, the models should be saved to the model repository.
        """
        pass
    

    @abstractmethod
    def new_data(self, data: Any) -> None:
        """
        Ingest new data into the training process.
        """
        pass

    def consume_status(self):
        self.set_status(TrainingStatus.READY)

    def get_status(self) -> TrainingStatus:
        """
        Get the current training status.
        
        Returns:
            TrainingStatus: Current status of the training process
        """
        with self._status_lock:
            return self.status


    def set_status(self, status: TrainingStatus) -> None:
        """
        Set the current training status.
        """
        with self._status_lock:
            self.status = status
