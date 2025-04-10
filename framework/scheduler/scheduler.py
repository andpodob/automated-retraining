"""
Scheduler class for coordinating model retraining workflow.
"""
import logging
import time
from typing import Any
import threading
import queue
from ..retraining.detector import Detector
from ..inference.inference import Inference
from ..trainer.trainer import Trainer, TrainingStatus
from ..data_source.data_source import DataSource, DataSourceStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrainingThread(threading.Thread):
    def __init__(self, trainer: Trainer, detector: Detector, data_queue: queue.Queue):
        super().__init__()
        self.trainer = trainer
        self.detector = detector
        self.data_queue = data_queue

    def run(self):
        print("RetrainingThread started")
        while True:
            try:
                while True:
                    data = self.data_queue.get(block=False)
                    self.trainer.new_data(data)
                    self.detector.new_data(data)
            except queue.Empty:
                logger.debug("Read all data from queue")

            if self.trainer.get_status() == TrainingStatus.TRAINING_IN_PROGRESS:
                logger.info("Waiting for training to be done before retraining")
                time.sleep(10)
                continue

            trainer_status = self.trainer.get_status()
            if self.detector.verdict() and trainer_status == TrainingStatus.READY:
                logger.info("Retraining needed")
                self.trainer.train()
            elif trainer_status in [TrainingStatus.TRAINING_IN_PROGRESS, TrainingStatus.GATHERING_DATA]:
                logger.info("Trainer is not ready to retrain")
            elif trainer_status == TrainingStatus.TRAINING_COMPLETED:
                logger.info("Training completed, result has not been consumed yet")
            elif trainer_status == TrainingStatus.TRAINING_FAILED:
                logger.info("Training failed, retraining...")
                self.trainer.train()
            else:
                logger.info("No retraining needed")
            time.sleep(10)


class InferenceThread(threading.Thread):
    def __init__(self, inference: Inference, data_queue: queue.Queue, inference_interval: int):
        super().__init__()
        self.inference = inference
        self.data_queue = data_queue
        self.inference_interval = inference_interval

    def run(self):
        while True:
            data = self.data_queue.get()
            if data is not None:
                self.inference.infer(data)
            time.sleep(self.inference_interval)

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
        self.trainer_data_queue = queue.Queue()
        self.inference_data_queue = queue.Queue()
        logger.info("Scheduler initialized with detector, inference, and trainer components")
    
    def run(self, datasource: DataSource, inference_interval: int = 10, infer_when_retraining: bool = True) -> None:
        """
        Run the retraining workflow.
        
        Args:
            new_data: List of new data points to analyze
        """
        logger.info("Starting retraining and inference threads")
        retraining_thread = RetrainingThread(self.trainer, self.detector, self.trainer_data_queue)
        inference_thread = InferenceThread(self.inference, self.inference_data_queue, inference_interval)
        retraining_thread.start()
        inference_thread.start()
        while True:
            if not retraining_thread.is_alive() or not inference_thread.is_alive():
                logger.info("Threads are not alive, exiting")
                break
            datasource_status = datasource.get_status()
            logger.info(f"Data source status: {datasource_status}")
            if datasource_status == DataSourceStatus.NO_NEW_DATA:
                logger.info("No new data available, skipping retraining")
                time.sleep(1)
                continue
            new_data = datasource.get_new_data()
            self.trainer_data_queue.put(new_data)
            self.inference_data_queue.put(new_data)
            # Wait before next iteration
            time.sleep(1)
