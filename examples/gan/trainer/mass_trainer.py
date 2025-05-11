import os
import pandas as pd
import numpy as np
import subprocess
import torch
import threading

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

MAX_THREADS = 15

SEQ_LEN = 90
OBSERVATION_LEN = 60

current_path = os.getcwd()
ROOT_DIR = os.path.join(current_path, ".mass_training")

def train_gan(exp_name: str, idx: int = 0):
    stdout_dir = os.path.join(ROOT_DIR, "stdout")
    if not os.path.exists(stdout_dir):
        os.makedirs(stdout_dir)
    stderr_dir = os.path.join(ROOT_DIR, "stderr")
    if not os.path.exists(stderr_dir):
        os.makedirs(stderr_dir)
    current_path = os.getcwd()
    script_path = os.path.dirname(os.path.abspath(__file__))
    cmd = f'python {os.path.join(script_path, "tts_gan", "train_gan.py")} \
    -gen_bs 16 \
    -dis_bs 16 \
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
    --training_set_path {os.path.join(ROOT_DIR, "data", f"chunk_{idx}", "training_set.pt")} \
    --test_set_path {os.path.join(ROOT_DIR, "data", f"chunk_{idx}", "validation_set.pt")} \
    --observation_size 60 \
    --max_epoch 1000 \
    --logs_dir {os.path.join(current_path, ".mass_training", "logs", f"{exp_name}_{idx}")}  \
    --random_seed 42 \
    --exp_name {exp_name}_{idx}'.split()

    stdout_file = open(os.path.join(stdout_dir, f"{idx}.txt"), "w")
    stderr_file = open(os.path.join(stderr_dir, f"{idx}.txt"), "w")
    # return subprocess.Popen(cmd)
    return subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)

class DataSet(Dataset):
    """
    DataSet is a class that contains the data for the training workflow.
    """
    def __init__(self, data):
        print(f"Initializing DataSet with {len(data)} samples")
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].reshape(1,1,-1), 1.0 

class TrainingWorkflowThread(threading.Thread):
    def __init__(self, exp_name: str, idx: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_name = exp_name
        self.idx = idx
        self.gan_process = None
        self.gan_status = False
    
    def run(self):
        print(f"Training workflow for chunk {self.idx} started!")
        self.gan_process = train_gan(self.exp_name, self.idx)
        status = self.gan_process.wait()
        if status != 0:
            self.gan_status = False
            return
        self.gan_status = True


def prepare_datasets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(os.path.dirname(current_dir), "data", "btcusd_1-min_data.csv")
    data = pd.read_csv(data_file)
    df = data[:SEQ_LEN*int(len(data)/SEQ_LEN)]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')
    df = df.dropna()
    df = df.drop(columns=['Volume', 'Open', 'High', 'Low'])
    df = df.astype('float32')
    samples = np.array_split(df['Close'].values, len(df)/SEQ_LEN)
    samples_transformed = []
    for sample in samples:
            X = sample.reshape(-1, 1)
            transformer = MinMaxScaler().fit(X[:OBSERVATION_LEN])
            transformed = transformer.transform(X)
            sample = transformed[:SEQ_LEN].squeeze()
            samples_transformed.append(sample)
    sample_batches = [samples_transformed[i:i + 1600] for i in range(0, len(samples_transformed), 1600)]
    
    for i, batch in enumerate(sample_batches):
        train, test = train_test_split(batch, test_size=0.3, shuffle=False)
        if not os.path.exists(os.path.join(ROOT_DIR, "data", f"chunk_{i}")):
            os.makedirs(os.path.join(ROOT_DIR, "data", f"chunk_{i}"))
        torch.save(train, os.path.join(ROOT_DIR, "data", f"chunk_{i}", "training_set.pt"))
        torch.save(test, os.path.join(ROOT_DIR, "data", f"chunk_{i}", "validation_set.pt"))


def start_training():
    print("Starting training!")
    idx = 0
    threads = []
    while os.path.exists(os.path.join(ROOT_DIR, "data", f"chunk_{idx}")):
        threads.append(TrainingWorkflowThread(exp_name="mass_training", idx=idx))
        idx += 1
    print(f"Prepared {len(threads)} chunks for training!")

    running_threads = []
    while threads:
        while len(running_threads) < MAX_THREADS and threads:
            thread = threads.pop(0)
            thread.start()
            running_threads.append(thread)
        for thread in running_threads: 
            if not thread.is_alive():
                thread.join()
                running_threads.remove(thread)
                if thread.gan_status:
                    print(f"Chunk {thread.idx} training finished successfully!")
                else:
                    print(f"Chunk {thread.idx} training failed!")

def main():
    # prepare_datasets()
    start_training()

if __name__ == "__main__":
    main()
    # dataset = DataSet(torch.load(os.path.join(ROOT_DIR, "chunk_0", "training_set.pt")))
    # print(f"Dataset length: {len(dataset)}")
    # print(f"First sample: {dataset[0]}")
