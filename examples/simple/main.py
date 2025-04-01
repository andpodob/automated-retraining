"""
Simple main script demonstrating basic functionality.
"""
from framework.scheduler.scheduler import Scheduler
from framework.retraining.detector import Detector
from framework.inference.inference import Inference
from framework.trainer.trainer import Trainer, TrainingStatus
from framework.data_source.data_source import DataSource, DataSourceStatus
from typing import Any
import random


class DummyInference(Inference):
    """
    Dummy inference class for testing.
    """
    def infer(self, input_data: Any) -> Any:
        print(f"Inferring with input: {input_data}")
        return input_data


class DummyDataSource(DataSource):
    """
    Dummy data source class for testing.
    """
    def get_new_data(self) -> Any:
        return "data"

    def get_status(self) -> DataSourceStatus:
        if random.random() < 0.8:
            return DataSourceStatus.NO_NEW_DATA
        else:
            return DataSourceStatus.NEW_DATA_AVAILABLE

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
    datasource = DummyDataSource()
    inference = DummyInference()
    trainer = DummyTrainer()
    detector = DummyDetector()
    scheduler = Scheduler(detector, inference, trainer)
    scheduler.run(datasource)


if __name__ == "__main__":
    main()
