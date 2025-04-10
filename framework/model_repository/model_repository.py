"""
Base class for implementing model storage and retrieval strategies.
"""

from abc import ABC, abstractmethod
from typing import Any


class ModelRepository(ABC):
    """
    Base class for implementing model storage and retrieval strategies.
    Concrete implementations should inherit from this class and implement the write and read methods.
    """
    
    @abstractmethod
    def write(self, model: Any, model_name: str, model_tag: str) -> None:
        """
        Save the model to the repository.
        
        Args:
            model: The model to save
        """
        pass
    
    @abstractmethod
    def load(self, model_name: str, model_tag: str) -> Any:
        """
        Load the model from the repository.
        
        Returns:
            Any: The loaded model
        """
        pass
    
    @abstractmethod
    def load_latest(self, model_name: str) -> Any:
        """
        Load the latest version of a model with the specified tag from the repository.
        
        Args:
            model_tag: Tag identifying the model to load
            
        Returns:
            Any: The latest version of the loaded model
        """
        pass 
