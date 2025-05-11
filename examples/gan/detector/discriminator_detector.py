"""
Detectors that triggers re-training on periodic basis.
"""
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from auto_retraining.retraining.detector import Detector
from auto_retraining.model_repository.pytorch.pytorch_model_repository import PytorchModelRepository

from sklearn.preprocessing import MinMaxScaler
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "trainer")) 
from tts_gan.gan_models import Discriminator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiscriminatorDetector(Detector):
    def __init__(self, model_repository: PytorchModelRepository, logs_path: str, min_samples: int = 1000, max_samples: int = 2000, sequence_length: int = 10, observation_size: int = 5):
        self.logs_path = logs_path
        self.logs_writer = SummaryWriter(logs_path)
        self.model_repository = model_repository    
        self.samples = []
        self.new_data_buffer = None
        self.current_verdict = False
        self.min_samples = min_samples
        self.sequence_length = sequence_length
        self.observation_size = observation_size
        self.model = None
        self.interation = 0
        self.max_samples = max_samples

    def new_data(self, data):
        if self.new_data_buffer is None:
            self.new_data_buffer = data
        else:
            self.new_data_buffer = pd.concat([self.new_data_buffer, data], ignore_index=True)

        if len(self.new_data_buffer) < self.sequence_length:
            return
        if len(data) == 0:
            return
        
        df = self.new_data_buffer[:self.sequence_length*int(len(self.new_data_buffer)/self.sequence_length)]
        self.new_data_buffer = self.new_data_buffer[self.sequence_length*int(len(self.new_data_buffer)/self.sequence_length):]
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df = df.set_index('Timestamp')
        df = df.dropna()
        df = df.drop(columns=['Volume', 'Open', 'High', 'Low'])
        df = df.astype('float32')
        samples = np.array_split(df['Close'].values, len(df)/self.sequence_length)
        for sample in samples:
            X = sample.reshape(-1, 1)
            transformer = MinMaxScaler().fit(X[:self.observation_size])
            transformed = transformer.transform(X)
            sample = transformed[:self.sequence_length].squeeze()
            self.samples.append(sample)
            self.samples = self.samples[-self.max_samples:]

        if len(self.samples) < self.min_samples:
            self.current_verdict = False    
            return
        
        if self.model is None:
            discriminator = Discriminator(seq_length=self.sequence_length) 
            discriminator.cuda()
            try:
                discriminator.load_state_dict(self.model_repository.load_latest("discriminator", device="cuda"))
            except FileNotFoundError:
                logger.warning("Discriminator model not found. At least one retraining is required.")
                self.current_verdict = True
                return
            self.model = discriminator
        
        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                samples = np.array(self.samples)
                samples = torch.tensor(samples, dtype=torch.float32).cuda()
                print(f"Samples shape: {samples.shape}, sequence_length: {self.sequence_length}, samples: {len(self.samples)}")
                samples = samples.reshape(len(self.samples), 1, 1, self.sequence_length)
                output = self.model(samples)
                output = output.cpu().numpy()
                output = np.mean(output, axis=0)
                self.logs_writer.add_scalar('detector_score', output, self.interation)
                self.interation += 1
                print(f"Discriminator output: {output}")
                if output < 0.5:
                    logger.info(f"DiscriminatorDetector verdict: retrain at {int(data['Timestamp'].values[0])}")
                    self.current_verdict = False
                else:
                    self.current_verdict = False


    
    def reset(self):
        self.current_verdict = False
        self.samples = []
        self.new_data_buffer = None

    def verdict(self):
        return self.current_verdict
