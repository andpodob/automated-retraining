"""
Base class for implementing model inference strategies.
"""

from abc import ABC, abstractmethod
from typing import Any


class Inference(ABC):
    """
    Base class for implementing model inference strategies.
    Concrete implementations should inherit from this class and implement the infer method.
    """
    
    @abstractmethod
    def infer(self, input_data: Any) -> Any:
        """
        Perform inference on the input data using the model.
        
        Args:
            input_data: Input data to perform inference on
            
        Returns:
            Any: Model predictions/inference results
        """
        pass 
