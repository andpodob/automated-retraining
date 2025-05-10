from abc import ABC, abstractmethod
from typing import Any
from enum import Enum


class DataSourceStatus(Enum):
    """Enumeration of possible data source status codes."""
    NO_NEW_DATA = "NO_NEW_DATA"
    NEW_DATA_AVAILABLE = "NEW_DATA_AVAILABLE"

class DataSource(ABC):
    """
    Abstract base class for data sources.
    """

    @abstractmethod
    def get_new_data(self) -> Any:
        """
        Get new data from the data source.
        """
        pass

    @abstractmethod
    def get_status(self) -> DataSourceStatus:
        """
        Get the status of the data source.
        """
        pass
