import pandas as pd
import time
import logging
from pathlib import Path
from typing import Any
from ..data_source import DataSource, DataSourceStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSVDataSource(DataSource):
    """
    A data source that reads data from a CSV file in batches.
    The entire file is loaded into memory during initialization.
    """
    
    def __init__(self, batch_size: int, interval: int, file: str):
        """
        Initialize the CSV data source.
        
        Args:
            batch_size: Number of records to return in each batch
            interval: Time interval in seconds between checks for new data
            file: Path to the CSV file
        """
        self.batch_size = batch_size
        self.interval = interval
        self.file = Path(file)
        self.last_check_time = 0
        self.status = DataSourceStatus.NO_NEW_DATA
        try:
            logger.info(f"Loading CSV file from {self.file}")
            self.data = pd.read_csv(self.file)
            self.current_position = 0
            self.total_rows = len(self.data)
            if self.total_rows > 0:
                self.status = DataSourceStatus.NEW_DATA_AVAILABLE
                logger.info(f"Successfully loaded {self.total_rows} rows from CSV file")
                logger.info(f"CSV columns: {list(self.data.columns)}")
            else:
                logger.warning("CSV file is empty")
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            self.data = pd.DataFrame()
            self.current_position = 0
            self.total_rows = 0
            self.status = DataSourceStatus.NO_NEW_DATA
        
    def get_new_data(self) -> Any:
        """
        Get the next batch of data from the in-memory DataFrame.
        
        Returns:
            A pandas DataFrame containing the next batch of records with original column names
        """
        if self.status != DataSourceStatus.NEW_DATA_AVAILABLE:
            logger.debug("No new data available")
            return None
            
        # Get the next batch of data
        end_position = min(self.current_position + self.batch_size, self.total_rows)
        batch = self.data.iloc[self.current_position:end_position].copy()
            
        if not batch.empty:
            logger.debug(f"Returning batch of {len(batch)} records (position {self.current_position} to {end_position})")
            self.current_position = end_position
            if self.current_position >= self.total_rows:
                logger.info("Reached end of data")
                self.status = DataSourceStatus.NO_NEW_DATA
            return batch
        else:
            logger.info("No more data available")
            self.status = DataSourceStatus.NO_NEW_DATA
            return None
            
    def get_status(self) -> DataSourceStatus:
        """
        Check if new data is available in the in-memory DataFrame.
        
        Returns:
            DataSourceStatus indicating if new data is available
        """
        current_time = time.time()
        
        # Only check for new data after the specified interval
        if current_time - self.last_check_time < self.interval:
            return self.status
            
        self.last_check_time = current_time
        
        # Check if we have more data to read
        if self.current_position < self.total_rows:
            self.status = DataSourceStatus.NEW_DATA_AVAILABLE
            logger.debug("New data available")
        else:
            self.status = DataSourceStatus.NO_NEW_DATA
            logger.debug("No new data available")
            
        return self.status
