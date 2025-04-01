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
    def detect(self, new_data: List[Any]) -> bool:
        """
        Detect if retraining is needed based on new data.
        
        Args:
            new_data: List of new data points to analyze
            
        Returns:
            bool: True if retraining is needed, False otherwise
        """
        pass
