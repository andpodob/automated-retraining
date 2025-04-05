"""
Scheduler class for coordinating model retraining workflow.
"""
import logging
import time
from typing import Any
from retraining.detector import Detector
from inference.inference import Inference
from trainer.trainer import Trainer, TrainingStatus
from data_source.data_source import DataSource, DataSourceStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Scheduler:
    """
    Coordinates the model retraining workflow using detector, inference, and trainer components.
    """
    
    def __init__(self, detector: Detector, inference: Inference, trainer: Trainer):
        """
        Initialize the scheduler with required components.
        
        Args:
            detector: Component responsible for detecting when retraining is needed
            inference: Component responsible for model inference
            trainer: Component responsible for model training
        """
        self.detector = detector
        self.inference = inference
        self.trainer = trainer
        logger.info("Scheduler initialized with detector, inference, and trainer components")
    
    def run(self, datasource: DataSource, inference_interval: int = 10, infer_when_retraining: bool = True) -> None:
        """
        Run the retraining workflow.
        
        Args:
            new_data: List of new data points to analyze
        """
        while True:
            # Wait before next iteration
            if inference_interval > 0:
                logger.info("Waiting for next iteration...")
                time.sleep(inference_interval)
            datasource_status = datasource.get_status()
            logger.info(f"Data source status: {datasource_status}")
            if datasource_status == DataSourceStatus.NO_NEW_DATA:
                logger.info("No new data available, skipping retraining")
                continue
            new_data = datasource.get_new_data()
            # Perform inference
            if infer_when_retraining or self.trainer.get_status() == TrainingStatus.TRAINING_DONE:
                logger.info("Triggering inference")
                self.inference.infer(new_data)
            
            # Ingest new data into the training process
            self.trainer.new_data(new_data)
            if self.trainer.get_status() == TrainingStatus.GATHERING_DATA:
                logger.info("Waiting for data to be gathered before retraining")
                continue
            
            if self.trainer.get_status() == TrainingStatus.TRAINING_IN_PROGRESS:
                logger.info("Waiting for training to be done before retraining")
                continue
            # Check if retraining is needed
            needs_retraining = self.detector.detect(new_data)
            logger.info(f"Detector result: {'Retraining needed' if needs_retraining else 'No retraining needed'}")
            
            if needs_retraining and self.trainer.get_status() == TrainingStatus.READY:
                logger.info("Starting model retraining process")
                self.trainer.train(new_data)
                logger.info(f"Training status: {self.trainer.get_status()}")
