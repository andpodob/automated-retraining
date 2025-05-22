import os
import argparse
import sys
import torch
import random
import copy
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from multiprocessing import Pool

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(os.path.join(parent_dir)) 
# print(sys.path)
# from gan_models import Generator


from sklearn.preprocessing import MinMaxScaler

from dtaidistance import dtw
from tqdm import tqdm 

import warnings
warnings.filterwarnings("ignore")
MAX_THREADS = 76

class DataSet(Dataset):
    """
    DataSet is a class that contains the data for the training workflow.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].reshape(1,1,-1), 1.0 


def get_best_matches(distance_matrix):
    matches = []
    for i in range(len(distance_matrix)):
        min_d = 10000000
        min_d_j = -1
        for j in range(len(distance_matrix[i])):
            if distance_matrix[i][j] < min_d:
                min_d = distance_matrix[i][j]
                min_d_j = j
        matches.append([i, min_d_j, min_d])
    distances = [x[2] for x in matches]

    return np.median(distances), matches

GLOBAL_COUNTER = 0
def compute_tasks(tasks):
    distances = []
    for (sig1, sig2, i, j) in tasks:
        distance = dtw.distance(sig1, sig2)
        distances.append((i, j, distance))
    return distances

def compute_task(task):
    (sig1, sig2, matrix, i, j) = task
    distance = dtw.distance(sig1, sig2)
    matrix[i][j] = distance



def similarity_matrix_parallel(signals_1, signals_2):
    matrix = np.zeros((len(signals_1), len(signals_2)))
    tasks = []
    for (i, sig1) in enumerate(signals_1):
        for (j, sig2) in enumerate(signals_2):
            tasks.append((sig1, sig2, i, j))
    n = 500
    tasks_batches = [tasks[i:i + n] for i in range(0, len(tasks), n)]
    with Pool (processes=MAX_THREADS) as pool:
        for result in pool.imap_unordered(compute_tasks, tasks_batches):
            for d in result:
                i, j, distance = d
                matrix[i][j] = distance
    return matrix

def similarity_matrix(signals_1, signals_2):
    matrix = np.zeros((len(signals_1), len(signals_2)))
    for (i, sig1) in enumerate(signals_1):
        for (j, sig2) in enumerate(signals_2):
            matrix[i][j] = dtw.distance(sig1.squeeze(), sig2.squeeze())
    return matrix


def prepare_signals_set(data):
    signals = []
    for signal in data:
        signal = signal.squeeze()
        scaler = MinMaxScaler()
        signal = signal.reshape(-1, 1)
        scaler.fit(signal)
        signals.append(scaler.transform(signal).reshape(1, -1))
    return signals


def datasets_distance(dataset_1, dataset_2, samples_count=10):
    samples_1 = dataset_1
    samples_2 = dataset_2
    if len(dataset_1) > samples_count:
        samples_1 = random.sample(dataset_1, samples_count)
        samples_2 = random.sample(dataset_2, samples_count)
    distance_matrix = similarity_matrix(samples_1, samples_2)
    avg_d, matches = get_best_matches(copy.deepcopy(distance_matrix))
    return avg_d


def convergence(training_set, seq_len, gen_net, N=100):
    # real_data = DataSet(torch.load(training_set_path))
    idxs = [i for i in range(len(training_set))]
    idxs = np.random.choice(idxs, N, replace=False)
    training_set = [training_set[i] for i in idxs]
    real_signals = prepare_signals_set(training_set)
    # checkpoint = torch.load(checkpoint_file, map_location='cpu')
    # gen_net = Generator(seq_len=seq_len)
    # gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
    gen_net.cpu()
    gen_net.eval()
    synthetic_data = [] 
    for i in range(N):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100)))
        fake_sigs = gen_net(fake_noise).to('cpu').detach().numpy()
        synthetic_data.append(fake_sigs)
    synthetic_signals = prepare_signals_set(synthetic_data)
    distance = datasets_distance(synthetic_signals, real_signals,  samples_count=N)
    return distance

def compute_partial_results(task):
    (i, j, dataset_1_path, dataset_2_path) = task
    dataset_1 = DataSet(torch.load(os.path.join(dataset_1_path, "training_set.pt")))
    dataset_2 = DataSet(torch.load(os.path.join(dataset_2_path, "training_set.pt")))
    distance = datasets_distance(dataset_1.data, dataset_2.data, samples_count=200)
    return (i, j, distance)
    

def collect_results(dataset_path_pattern, datasets_count, results_path):
    results = []
    tasks = []
    for i in range(datasets_count):
        for j in range(i+1, datasets_count):
            dataset_1_path = os.path.join(dataset_path_pattern, f"chunk_{i}")
            dataset_2_path = os.path.join(dataset_path_pattern, f"chunk_{j}")
            tasks.append((i, j, dataset_1_path, dataset_2_path))
    with Pool(processes=MAX_THREADS) as pool:
        for result in tqdm(pool.imap_unordered(compute_partial_results, tasks), total=len(tasks)):
            results.append(result)

    df = pd.DataFrame(results, columns =['dataset_1', 'dataset_2', 'distance'])
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    df.to_csv(os.path.join(results_path, "results.csv"), index=False)

def main2():
    exp_path = os.path.join('logs', f'crypto-{0}')
    checkpoint_file = os.path.join(exp_path, 'Model', 'checkpoint')
    conv = convergence(0, 1440, 480, 30, checkpoint_file)
    print(conv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_path', default="checkpoints", type=str,
                    help='checkpoints path')
    parser.add_argument('--starting_epoch', default=0, type=int, help='epoch for which to start exporting')
    parser.add_argument('--compare_datasets', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--evaluate_experiment_models', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--dataset_path_pattern', default="", type=str,
                    help='')
    parser.add_argument('--datasets_count', default=1, type=int,help='number of datasets to compare')
    parser.add_argument('--results_path', default="", type=str,help='')

    args = parser.parse_args()
    if args.compare_datasets:
        if args.dataset_path_pattern == "":
            print("Dataset path pattern is empty.")
            sys.exit(1)
        if args.results_path == "":
            print("Results path is empty.")
            sys.exit(1)
        if args.datasets_count < 1:
            print("Datasets count is less than 1.")
            sys.exit(1)
        collect_results(args.dataset_path_pattern, args.datasets_count, args.results_path)
        sys.exit(0)

    filenames = os.listdir(args.checkpoints_path)
    data_org = crypto("./data/btcusd_5_min_by_day_scale.csv")
    signals_org = prepare_signals_set(data_org)[-100:]
    for i in range(args.starting_epoch, 100000, 50):
        if f'checkpoint_{i}' in filenames:
            print(f"Calculating coherence for epoch {i}.")
            checkpoint_file = os.path.join(args.checkpoints_path, f'checkpoint_{i}')
            checkpoint = torch.load(checkpoint_file)
            writer = SummaryWriter(checkpoint["path_helper"]['log_path'])
            data_checkpoint = syn_crypto(N, checkpoint_file)
            signals_checkpoint = prepare_signals_set(data_checkpoint)
            distance_matrix = similarity_matrix(signals_checkpoint, signals_org)
            avg_d, matches = get_best_matches(copy.deepcopy(distance_matrix))
            writer.add_scalar('avg_distance_v2', avg_d, i)


if __name__ == '__main__':
    main()
