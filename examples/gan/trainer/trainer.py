"""
Trainer workflow implementation that follows the framework's Trainer interface.
"""
import logging
import os
import sys
import torch
import threading
import datetime
import random
import subprocess
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Any, Optional
from framework.trainer.trainer import Trainer, TrainingStatus
from framework.model_repository.pytorch.pytorch_model_repository import PytorchModelRepository
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from tts_gan.gan_models import Generator, Discriminator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def remove_random_elements(arr, n):
  """Removes n random elements from a list.

  Args:
    arr: The list to remove elements from.
    n: The number of random elements to remove.

  Returns:
    A new list with n random elements removed.
  """
  if n >= len(arr):
    return []  # Or return an empty list if you prefer

  indices_to_remove = random.sample(range(len(arr)), n)
  new_arr = [item for i, item in enumerate(arr) if i not in indices_to_remove]
  return new_arr

def train_lstm():
    current_path = os.getcwd()
    script_path = os.path.dirname(os.path.abspath(__file__))
    cmd = f'python {os.path.join(script_path, "lstm", "train_lstm.py")} \
    --exp_name lstm \
    --seq_len 90 \
    --val_set_ratio 0.2 \
    --prediction_size 30 \
    --seed 42 \
    --training_set_path {os.path.join(current_path, ".training", "augmented_training_set.pt")} \
    --test_set_path {os.path.join(current_path, ".training", "validation_set.pt")} \
    --logs_dir {os.path.join(current_path, ".training", "logs", "lstm")} \
    --epochs 100'.split()
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def train_gan():
    current_path = os.getcwd()
    script_path = os.path.dirname(os.path.abspath(__file__))
    cmd = f'python {os.path.join(script_path, "tts_gan", "train_gan.py")} \
    -gen_bs 16 \
    -dis_bs 16 \
    --load_path {os.path.join(current_path, ".training", "logs", "gan", "Model", "checkpoint")} \
    --rank 0 \
    --world-size 1 \
    --bottom_width 8 \
    --max_iter 500000 \
    --img_size 32 \
    --gen_model my_gen \
    --dis_model my_dis \
    --df_dim 384 \
    --d_heads 4 \
    --d_depth 3 \
    --g_depth 5,4,2 \
    --dropout 0 \
    --latent_dim 100 \
    --gf_dim 1024 \
    --num_workers 8 \
    --g_lr 0.0001 \
    --d_lr 0.0003 \
    --optimizer adam \
    --loss lsgan \
    --wd 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --phi 1 \
    --batch_size 16 \
    --num_eval_imgs 50000 \
    --init_type xavier_uniform \
    --n_critic 1 \
    --val_freq 20 \
    --print_freq 50 \
    --grow_steps 0 0 \
    --fade_in 0 \
    --ema_kimg 500 \
    --ema_warmup 0.1 \
    --ema 0.9999 \
    --diff_aug translation,cutout,color \
    --seq_len 90 \
    --training_set_path {os.path.join(current_path, ".training", "training_set.pt")} \
    --test_set_path {os.path.join(current_path, ".training", "validation_set.pt")} \
    --observation_size 60 \
    --max_epoch 3 \
    --logs_dir {os.path.join(current_path, ".training", "logs", "gan")}  \
    --discriminator_path {os.path.join(current_path, ".training", "gan", "discriminator.pth")} \
    --generator_path {os.path.join(current_path, ".training", "gan", "generator.pth")} \
    --random_seed 42 \
    --exp_name gan'.split()
    # return subprocess.Popen(cmd)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class DataSet(Dataset):
    """
    DataSet is a class that contains the data for the training workflow.
    """
    def __init__(self, data: Any):
        print(f"Initializing DataSet with {len(data)} samples")
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].reshape(1,1,-1), 1.0


class DataSets:
    """
    DataSets is a class that contains the data sets for the training workflow.
    """
    def __init__(self, min_samples: int, max_samples: int, split_ratio: float, sequence_length: int, observation_size: int): 
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.split_ratio = split_ratio
        self.new_data_buffer = None
        self.training_set = []
        self.validation_set = []
        self.observation_size = observation_size
        self.sequence_length = sequence_length

    def ingest_data(self, data: Any) -> None:
        """
        Ingests data from a data source to the training set.
        """
        if self.new_data_buffer is None:
            self.new_data_buffer = data
        else:
            self.new_data_buffer = pd.concat([self.new_data_buffer, data], ignore_index=True)

        if len(self.new_data_buffer) < self.sequence_length:
            return

        df = self.new_data_buffer[:self.sequence_length*int(len(self.new_data_buffer)/self.sequence_length)]
        self.new_data_buffer = self.new_data_buffer[self.sequence_length*int(len(self.new_data_buffer)/self.sequence_length):]
        # Process the new data buffer
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
            if random.random() < self.split_ratio:
                self.validation_set.append(sample)
            else:
                self.training_set.append(sample)

        train_samples_to_remove = len(self.training_set) - int((self.max_samples * (1 - self.split_ratio)))
        if train_samples_to_remove > 0:
            self.training_set = remove_random_elements(self.training_set, train_samples_to_remove)

        validation_samples_to_remove = len(self.validation_set) - int(self.max_samples*self.split_ratio)
        if validation_samples_to_remove > 0:
            self.validation_set = remove_random_elements(self.validation_set, validation_samples_to_remove)


    def get_training_set_size(self) -> int:
        """
        Returns the size of the training set.
        """
        return len(self.training_set)
    
    def get_training_set(self) -> Any:
        """
        Returns the training set.
        """
        return DataSet(self.training_set)
    
    def get_validation_set(self) -> Any:
        """
        Returns the validation set.
        """
        return DataSet(self.validation_set)
    

class TrainingWorkflowThread(threading.Thread):
    """
    GanTrainingWorkflowThread is a class that contains the training workflow.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gan_process = None
        self.gan_status = None
        self.lstm_status = None
        self.model_repository = PytorchModelRepository()


    def save_augmented_data_set(self, augmentation_factor: int) -> None:
        """
        Save augmented data set to disk so the training actual process can load it.
        """
        current_path = os.getcwd()
        root_dir = os.path.join(current_path, ".training")
        raw_training_data = torch.load(os.path.join(root_dir, "training_set.pt"))
        generator = Generator(seq_len=90) 
        generator.cuda()
        generator.load_state_dict(self.model_repository.load_latest("generator", device="cuda"))
        augmented_data = []
        augmented_samples_count = len(raw_training_data) * augmentation_factor
        logger.info(f"Generating augmented dataset")
        for i in range(augmented_samples_count):
            fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100))).to('cuda')
            fake_sigs = generator(fake_noise.to(torch.float32)).to('cpu').detach().numpy()
            augmented_data.append(fake_sigs.squeeze())
        augmented_data = raw_training_data + augmented_data
        logger.info(f"Saving augmented dataset of size: {len(augmented_data)}")
        torch.save(augmented_data, os.path.join(root_dir, "augmented_training_set.pt"))


    def run(self):
        self.gan_process = train_gan()
        status = self.gan_process.wait()
        if status != 0:
            self.gan_status = False
            return
        self.gan_status = True
        self.save_augmented_data_set(2)
        self.lstm_process = train_lstm()
        status = self.lstm_process.wait()
        if status != 0:
            self.lstm_status = False
        self.lstm_status = True


class LstmTrainerWithGanAugmentation(Trainer):
    """
    Trainer runs a training workflow that:
    1. Ingests data from a data source to the training set.
    2. Trains GAN model
    3. Trains Forecasting model on augmented data.
    """

    def __init__(self, min_samples: int, observation_size: int, sequence_length: int):
        """
        Initialize the trainer.
        """
        super().__init__()
        self.data_sets = DataSets(min_samples=min_samples, max_samples=2000, split_ratio=0.2, sequence_length=sequence_length, observation_size=observation_size)
        self.model_repository = PytorchModelRepository()
        logger.info(f"Initialized GAN Trainer")


    def save_data_sets(self) -> None:
        """
        Save training and validation sets to disk so the training actual process can load it.
        """
        current_path = os.getcwd()
        root_dir = os.path.join(current_path, ".training")
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        torch.save(self.data_sets.get_training_set().data, os.path.join(root_dir, "training_set.pt"))
        torch.save(self.data_sets.get_validation_set().data, os.path.join(root_dir, "validation_set.pt"))
        

    def train(self) -> None:
        """
        Run training workflow.
        1. Save training and validation sets to disk so the training actual process can load it.
        2. Trigger GAN training process.
        3. Trigger Forecasting model training process.
        """
        self.save_data_sets()
        super().set_status(TrainingStatus.TRAINING_IN_PROGRESS)
        self.training_workflow_thread = TrainingWorkflowThread()
        self.training_workflow_thread.start()


    def update_status(self, current_status: TrainingStatus) -> None:
        """
        Get the current training status.
        """
        if current_status == TrainingStatus.TRAINING_IN_PROGRESS:
            if not self.training_workflow_thread.is_alive():
                if self.training_workflow_thread.gan_status and self.training_workflow_thread.lstm_status:
                    super().set_status(TrainingStatus.TRAINING_COMPLETED)
                else:
                    super().set_status(TrainingStatus.TRAINING_FAILED)


    def get_status(self) -> TrainingStatus:
        """
        Get the current training status and update it if needed.
        
        Returns:
            TrainingStatus: Current status of the training process
        """
        self.update_status(super().get_status())
        return super().get_status()


    def new_data(self, data: Any) -> None:
        """
        Ingest new data into the training process.
        """
        self.data_sets.ingest_data(data)
        if self.data_sets.get_training_set_size() >= self.data_sets.min_samples and self.get_status() == TrainingStatus.GATHERING_DATA:
            super().set_status(TrainingStatus.READY)


if __name__ == "__main__":
    print("Training GAN")
    p = train_gan()
    p.wait()
    # print("Training LSTM")
    # p = train_lstm()
    # p.wait()
