import argparse
import sys
import os
import pandas as pd
import numpy as np
import subprocess
import torch
import threading

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, "trainer")) 
from tts_gan.gan_models import Generator
from tts_gan.convergence import convergence

MAX_THREADS = 15

SEQ_LEN = 90
OBSERVATION_LEN = 60

current_path = os.getcwd()
ROOT_DIR = os.path.join(current_path, ".mass_training")
FILE = __file__

# 1
    # embed_dim_g = [5, 15]
    # embed_dim_d = [5, 15]
    # g_num_heads = [3, 5]
    # d_num_heads = [3, 5]
    # d_depth = [1, 3]
    # g_depth = [1, 3]
    # g_patch_size = [5, 15]
    # d_patch_size = [5, 15]
    # 2
    # embed_dim_g = [5]
    # embed_dim_d = [10, 20]
    # g_num_heads = [5, 10]
    # d_num_heads = [5, 10]
    # d_depth = [3]
    # g_depth = [1]
    # g_patch_size = [10, 30]
    # d_patch_size = [5]
    # 3
    # embed_dim_g = [5]
    # embed_dim_d = [20, 30, 40, 50, 60]
    # g_num_heads = [5]
    # d_num_heads = [10, 20]
    # d_depth = [3]
    # g_depth = [1]
    # g_patch_size = [30]
    # d_patch_size = [5]
    # 4
    # embed_dim_g = [5]
    # embed_dim_d = [20, 30, 40, 50, 60, 80]
    # g_num_heads = [5]
    # d_num_heads = [10, 20, 40]
    # d_depth = [3]
    # g_depth = [1]
    # g_patch_size = [30]
    # d_patch_size = [5]
    # 5
    # embed_dim_g = [5, 15, 30]
    # embed_dim_d = [80]
    # g_num_heads = [5, 15, 30]
    # d_num_heads = [40]
    # d_depth = [3]
    # g_depth = [1]
    # g_patch_size = [30]
    # d_patch_size = [5]
    # 7
    # embed_dim_d = [80, 90,120]
    # d_num_heads = [30, 40]
    # d_depth = [3]
    # d_patch_size = [5]

    # g_depth = [1]
    # embed_dim_g = [5]
    # g_num_heads = [5]
    # g_patch_size = [30]

    # 8
    # embed_dim_d = [20, 30, 80, 90, 120]
    # d_num_heads = [10, 30]
    # d_depth = [3, 4, 5]
    # d_patch_size = [5, 15]

    # g_depth = [1]
    # embed_dim_g = [5]
    # g_num_heads = [5]
    # g_patch_size = [30]

    # LARGE
embed_dim_d = [120, 240]
d_num_heads = [30, 60]
d_depth = [3]
d_patch_size = [15]

g_depth = [2, 3]
embed_dim_g = [3, 5]
g_num_heads = [3, 5]
g_patch_size = [10]

def train_gan(exp_name: str, data_path: str, idx: int, config):
    stdout_dir = os.path.join(ROOT_DIR, exp_name, "stdout")
    stderr_dir = os.path.join(ROOT_DIR, exp_name, "stderr")
    script_path = os.path.dirname(os.path.abspath(FILE))
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
    --g_embed_dim {config["g_embed_dim"]} \
    --d_embed_dim {config["d_embed_dim"]} \
    --g_num_heads {config["g_num_heads"]} \
    --d_num_heads {config["d_num_heads"]} \
    --d_depth {config["d_depth"]} \
    --g_depth {config["g_depth"]} \
    --g_patch_size {config["g_patch_size"]} \
    --d_patch_size {config["d_patch_size"]} \
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
    --training_set_path {os.path.join(data_path, "training_set.pt")} \
    --test_set_path {os.path.join(data_path, "validation_set.pt")} \
    --observation_size 60 \
    --max_epoch 500 \
    --logs_dir {os.path.join(ROOT_DIR, exp_name,  "logs", f"{exp_name}_{idx}")}  \
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
    def __init__(self, exp_name: str, data_path: str, idx: int, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_name = exp_name
        self.idx = idx
        self.gan_process = None
        self.gan_status = False
        self.config = config
        self.data_path = data_path
    
    def run(self):
        print(f"Training workflow for with id {self.idx} started!")
        print(self.config)
        self.gan_process = train_gan(self.exp_name, self.data_path, self.idx, self.config)
        status = self.gan_process.wait()
        if status != 0:
            self.gan_status = False
            return
        self.gan_status = True


def start_training(exp_name):
    print("Starting training!")
    idx = 0
    threads = []
    config = {
        "g_embed_dim": 3,
        "d_embed_dim": 240,
        "g_num_heads": 3,
        "d_num_heads": 60,
        "d_depth": 3,
        "g_depth": 3,
        "g_patch_size": 10,
        "d_patch_size": 15
    }
    stdout_dir = os.path.join(ROOT_DIR, exp_name, "stdout")
    if not os.path.exists(stdout_dir):
        os.makedirs(stdout_dir)
    stderr_dir = os.path.join(ROOT_DIR, exp_name, "stderr")
    if not os.path.exists(stderr_dir):
        os.makedirs(stderr_dir)
    while os.path.exists(os.path.join(ROOT_DIR, exp_name, "data", f"chunk_{idx}")):
        threads.append(TrainingWorkflowThread(exp_name=exp_name, data_path=os.path.join(ROOT_DIR, exp_name, "data", f"chunk_{idx}"), idx=idx, config=config))
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

def start_optimization(exp_name):
    cofnigs = []
    for g_embed_dim in embed_dim_g:
        for d_embed_dim in embed_dim_d:
            for g_num_head in g_num_heads:
                for d_num_head in d_num_heads:
                    for d_d in d_depth:
                        for g_d in g_depth:
                            for g_p in g_patch_size:
                                for d_p in d_patch_size:
                                    if g_embed_dim % g_num_head != 0 or d_embed_dim % d_num_head != 0:
                                        continue
                                    cofnigs.append({
                                        "g_embed_dim": g_embed_dim,
                                        "d_embed_dim": d_embed_dim,
                                        "g_num_heads": g_num_head,
                                        "d_num_heads": d_num_head,
                                        "d_depth": d_d,
                                        "g_depth": g_d,
                                        "g_patch_size": g_p,
                                        "d_patch_size": d_p
                                    })
    print(f"Prepared {len(cofnigs)} configurations for optimization!")
    threads = []
    for (i, config) in enumerate(cofnigs):
        threads.append(TrainingWorkflowThread(exp_name=exp_name, data_path=os.path.join(ROOT_DIR, exp_name, "data", "chunk_0"), idx=i, config=config))
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
                    print(f"Cofnig {thread.config} training finished successfully!")
                else:
                    print(f"Cofnig {thread.config} training failed!")

def convergence_test(exp_name: str):
    embed_dim_d = [120]
    d_num_heads = [30]
    d_depth = [4]
    d_patch_size = [15]

    g_depth = [1, 3, 5]
    embed_dim_g = [5, 10, 20, 90]
    g_num_heads = [5, 10, 20, 30]
    g_patch_size = [10, 30, 45]
    model_cofnigs = []
    for g_embed_dim in embed_dim_g:
        for d_embed_dim in embed_dim_d:
            for g_num_head in g_num_heads:
                for d_num_head in d_num_heads:
                    for d_d in d_depth:
                        for g_d in g_depth:
                            for g_p in g_patch_size:
                                for d_p in d_patch_size:
                                    if g_embed_dim % g_num_head != 0 or d_embed_dim % d_num_head != 0:
                                        continue
                                    model_cofnigs.append({
                                        "g_embed_dim": g_embed_dim,
                                        "d_embed_dim": d_embed_dim,
                                        "g_num_heads": g_num_head,
                                        "d_num_heads": d_num_head,
                                        "d_depth": d_d,
                                        "g_depth": g_d,
                                        "g_patch_size": g_p,
                                        "d_patch_size": d_p
                                    })
    data_path = os.path.join(ROOT_DIR, exp_name, "data", "chunk_0")
    data = torch.load(os.path.join(data_path, "training_set.pt"))
    for (i, config) in enumerate(model_cofnigs):
        print("config", config) 
        gen_net = Generator(seq_len=90, embed_dim=config["g_embed_dim"], patch_size=config["g_patch_size"], num_heads=config["g_num_heads"], depth=config["g_depth"])
        gen_net.cuda()
        model_paths = []
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(ROOT_DIR, exp_name, "logs", f"{exp_name}_{i}")):
            for dirname in dirnames:
                if os.path.exists(os.path.join(dirpath, dirname, "Model", "checkpoint")):
                    model_paths.append(os.path.join(dirpath, dirname, "Model", "checkpoint"))
        for model_path in model_paths:
            checkpoint = torch.load(model_path, map_location='cuda')
            try:
                gen_net.load_state_dict(checkpoint['gen_state_dict'])
                gen_net.eval()
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
            conv = convergence(data, SEQ_LEN, gen_net, N=300, M=300)
            print(f"Convergence for model {model_path} is {conv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default="", type=str,
                    help='experiment name')
    parser.add_argument('--optimize', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--convergence_test', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.optimize:
        start_optimization(exp_name=args.exp_name)
    elif args.convergence_test:
        convergence_test(exp_name=args.exp_name)
    else:
        start_training(exp_name=args.exp_name)

if __name__ == "__main__":
    main()
