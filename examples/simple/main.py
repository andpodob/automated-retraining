"""
Simple main script demonstrating basic functionality.
"""
import os
from framework.scheduler.scheduler import Scheduler
from framework.retraining.detector import Detector
from framework.inference.inference import Inference
from framework.trainer.trainer import Trainer, TrainingStatus
from framework.data_source.csv.csv_datasource import CSVDataSource
from typing import Any
import random


class DummyInference(Inference):
    """
    Dummy inference class for testing.
    """
    def infer(self, input_data: Any) -> Any:
        print(f"Inferring with input: {input_data}")
        return input_data


class DummyTrainer(Trainer):
    """
    Dummy trainer class for testing.
    """
    def train(self, input_data: Any) -> Any:
        print(f"Training with input: {input_data}")
        return input_data

    def get_status(self) -> TrainingStatus:
        if random.random() < 0.2:
            return TrainingStatus.TRAINING_DONE
        else:
            return TrainingStatus.TRAINING_IN_PROGRESS


class DummyDetector(Detector):
    """
    Dummy detector class for testing.
    """
    def detect(self, input_data: Any) -> bool:
        print(f"Detecting with input: {input_data}")
        return random.random() < 0.4


def main():
    """Main entry point of the script."""
    # Get the absolute path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "data", "btcusd_1-min_data.csv")
    
    datasource = CSVDataSource(batch_size=10, interval=1, file=data_file)
    inference = DummyInference()
    trainer = DummyTrainer()
    detector = DummyDetector()
    scheduler = Scheduler(detector, inference, trainer)
    scheduler.run(datasource)


if __name__ == "__main__":
    main()
