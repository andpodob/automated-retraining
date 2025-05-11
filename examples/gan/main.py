"""
Example of using the framework to re-train a forecasting model with data augmented 
by a GAN and GAN Discriminator used as a detector.
"""
import argparse
import os
from detector.periodic_detector import PeriodicDetector
from detector.discriminator_detector import DiscriminatorDetector
from trainer.trainer import LstmTrainerWithGanAugmentation
# from lstm.model import LSTMModel
from auto_retraining.scheduler.scheduler import Scheduler
from auto_retraining.retraining.detector import Detector
from auto_retraining.inference.inference import Inference, DataAdapter, OutputSink
from auto_retraining.inference.pytorch.pytorch_inference import PyTorchInference
from auto_retraining.trainer.trainer import Trainer, TrainingStatus
from auto_retraining.data_source.csv.csv_datasource import CSVDataSource
from auto_retraining.model_repository.pytorch.pytorch_model_repository import PytorchModelRepository
from typing import Any
import torch
import pandas as pd


class TestingSink(OutputSink):
    """
    Output sink class for testing.
    """
    def receive(self, output: Any) -> None:
        pass
        # print(f"Output: <inference output>")

class DummyInference(Inference):
    def __init__(self, output_sink: OutputSink):
        self.output_sink = output_sink

    def infer(self, input_data: Any) -> Any:
        self.output_sink.receive({"input": input_data, "output": input_data})

class DummyTrainer(Trainer):
    """
    Dummy trainer class for testing.
    """
    def train(self, input_data: Any) -> None:
        print(f"Skiping training")

    def get_status(self) -> TrainingStatus:
        return TrainingStatus.TRAINING_DONE


class DummyDetector(Detector):
    """
    Dummy detector class for testing.
    """

    def new_data(self, new_data: Any) -> None:
        pass

    def verdict(self) -> bool:
        return True

class LstmDataAdapter(DataAdapter):
    """
    Data adapter class for LSTM model.
    """
    def __init__(self, observarion_len: int):
        self.observarion_len = observarion_len
    
    def transform(self, input_data: Any) -> Any:
        input_data['Timestamp'] = pd.to_datetime(input_data['Timestamp'], unit='s')
        input_data = input_data.set_index('Timestamp')
        input_data = input_data.dropna()
        input_data = input_data.drop(columns=['Volume', 'Open', 'High', 'Low'])
        input_data = input_data.iloc[-self.observarion_len:]
        tensor = torch.tensor(input_data.values, dtype=torch.float32)
        tensor = tensor.reshape(1, -1, 1)
        return tensor

def main():
    """Main entry point of the script."""
    # Get the absolute path to the data file
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Name of experiment')
    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "data", "btcusd_1-min_data.csv")
    datasource = CSVDataSource(batch_size=9000, interval=0, file=data_file)
    # lstm = LSTMModel(input_dim=1, output_dim=15, layer_dim=3, hidden_dim=64, dropout_prob=0.2)
    # model_checkpoint = torch.load(os.path.join(current_dir, "models", "lstm_model.pth"), map_location=torch.device('cpu'))
    # lstm.load_state_dict(model_checkpoint["lstm_state_dict"])
    # inference = PyTorchInference(model=lstm, data_adapter=LstmDataAdapter(observarion_len=30), output_sink=TestingSink())
    trainer = LstmTrainerWithGanAugmentation(exp_name=args.exp_name, min_samples=1000, observation_size=30, sequence_length=90)
    # detector = PeriodicDetector(100)
    current_path = os.getcwd()
    detector = DiscriminatorDetector(model_repository=PytorchModelRepository(experiment_name=args.exp_name), logs_path=os.path.join(current_path, ".training", "logs", args.exp_name, "detector"), min_samples=1000, observation_size=30, sequence_length=90)
    scheduler = Scheduler(detector, DummyInference(output_sink=TestingSink()), trainer)
    scheduler.run(datasource, inference_interval=0, test_mode=True)


if __name__ == "__main__":
    main()
