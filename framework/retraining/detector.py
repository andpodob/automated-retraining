"""
Base class for implementing retraining detection strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Any


class Detector(ABC):
    """
    Base class for implementing retraining detection strategies.
    Concrete implementations should inherit from this class and implement the detect method.
    """
    
    @abstractmethod
    def new_data(self, new_data: Any) -> None:
        """
        Adds data to the detector.
        
        Args:
            new_data: List of new data points to analyze
            
        Returns:
            bool: True if retraining is needed, False otherwise
        """
        pass

    @abstractmethod
    def verdict(self) -> bool:
        """
        Returns the verdict of the detector based on the data added to the detector.
        """
        pass
