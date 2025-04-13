import os
import argparse
import sys
import torch
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from gan_models import Generator
# Add parent directory to the system path to access trainer.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from trainer import DataSet

from sklearn.preprocessing import MinMaxScaler

from dtaidistance import dtw
from tqdm import tqdm 



def get_best_matches(distance_matrix):
    matches = []
    average = 0
    for i in range(len(distance_matrix)):
        min_d = 10000000
        min_d_j = -1
        for j in range(len(distance_matrix[i])):
            if distance_matrix[i][j] < min_d:
                min_d = distance_matrix[i][j]
                min_d_j = j
        matches.append([i, min_d_j, min_d])
        for x in range(len(distance_matrix)):
            distance_matrix[x][min_d_j] = 10000000
        average = (average * i + min_d) / (i + 1) 

    return average, matches

def similarity_matrix(signals_1, signals_2):
    matrix = np.zeros((len(signals_1), len(signals_2)))
    for (i, sig1) in tqdm(enumerate(signals_1), total=len(signals_1)):
        for (j, sig2) in enumerate(signals_2):
            matrix[i][j] = dtw.distance(sig1[0], sig2[0])

    return matrix

def prepare_signals_set(data):
    signals = []
    for signal in data:
        scaler = MinMaxScaler()
        signal = signal[0][0].reshape(-1, 1)
        scaler.fit(signal)
        signals.append(scaler.transform(signal).reshape(1, -1))
    return signals

def convergence(training_set_path, seq_len, checkpoint_file, N=100, M=20):
    real_data = DataSet(torch.load(training_set_path))
    real_signals = prepare_signals_set(real_data)[:N]
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    gen_net = Generator(seq_len=seq_len)
    gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
    synthetic_data = [] 
    for i in range(10):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100)))
        fake_sigs = gen_net(fake_noise).to('cpu').detach().numpy()
        synthetic_data.append(fake_sigs)
    synthetic_signals = prepare_signals_set(synthetic_data)[:M]
    distance_matrix = similarity_matrix(synthetic_signals, real_signals)
    avg_d, matches = get_best_matches(copy.deepcopy(distance_matrix))
    return avg_d


def main2():
    exp_path = os.path.join('logs', f'crypto-{0}')
    checkpoint_file = os.path.join(exp_path, 'Model', 'checkpoint')
    conv = convergence(0, 1440, 480, 30, checkpoint_file)
    print(conv)

if __name__ == '__main__':
    main2()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_path', default="checkpoints", type=str,
                    help='checkpoints path')
    parser.add_argument('--starting_epoch', default=0, type=int, help='epoch for which to start exporting')
    args = parser.parse_args()
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
