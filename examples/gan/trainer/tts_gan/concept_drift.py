import argparse
import numpy as np
import os
import torch
import pandas as pd

from multiprocessing import Pool
from gan_models import Discriminator
from tqdm import tqdm 

import warnings
warnings.filterwarnings("ignore")

MAX_THREADS = 3
SEQUENCE_LENGTH = 90
CONFIG = {
        "g_embed_dim": 3,
        "d_embed_dim": 240,
        "g_num_heads": 3,
        "d_num_heads": 60,
        "d_depth": 3,
        "g_depth": 3,
        "g_patch_size": 10,
        "d_patch_size": 15
    }


def concept_drifts(exp_name, model_repo_path, model_id, dataset_path_pattern, datasets_count):
    model = Discriminator(seq_length=SEQUENCE_LENGTH, patch_size=CONFIG["d_patch_size"], emb_size=CONFIG["d_embed_dim"], depth=CONFIG["d_depth"], num_heads=CONFIG["d_num_heads"])
    model_path = os.path.join(model_repo_path, f"{exp_name}_{model_id}", "discriminator", "latest.pt")
    print("I'm here")
    if not os.path.exists(model_path):
        return [(model_id, i, -1) for i in range(datasets_count)]
    print("I'm NOT here")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.cuda()
    model.eval()
    results = []
    for i in range(model_id+1, datasets_count):
        dataset_path = os.path.join(dataset_path_pattern, f"chunk_{i}")
        samples = torch.load(os.path.join(dataset_path, "training_set.pt"), map_location=torch.device('cuda'))
        samples = np.array(samples)
        print(len(samples))
        samples = torch.tensor(samples, dtype=torch.float32).cuda()
        samples = samples.reshape(len(samples), 1, 1, SEQUENCE_LENGTH)
        output = model(samples)
        output = output.detach().cpu().numpy()
        output = 1-np.mean(output, axis=0)
        results.append((model_id, i, output[0]))    
    return results


def computer_partial_result(task):
    (exp_name, model_path, model_id, dataset_path_pattern, datasets_count) = task
    return concept_drifts(exp_name, model_path, model_id, dataset_path_pattern, datasets_count)

def collect_results(exp_name, model_repo_path, dataset_path_pattern, datasets_count, results_path):
    results = []
    tasks = []
    for i in range(datasets_count):
        tasks.append((exp_name, model_repo_path, i, dataset_path_pattern, datasets_count))

    with Pool(processes=MAX_THREADS) as pool:
        for result in tqdm(pool.imap_unordered(computer_partial_result, tasks), total=len(tasks)):
            results.extend(result)
    
    df = pd.DataFrame(results, columns =['dataset_1', 'dataset_2', 'distance'])
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    df.to_csv(os.path.join(results_path, "drift_results.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_repo_path', default="", type=str,
                    help='model repository path')
    parser.add_argument('--dataset_path_pattern', default="", type=str,
                    help='')
    parser.add_argument('--exp_name', default="", type=str,
                    help='')
    parser.add_argument('--datasets_count', default=1, type=int,help='number of datasets to compare')
    parser.add_argument('--results_path', default="", type=str,help='')

    args = parser.parse_args()
    if args.dataset_path_pattern == "":
        raise ValueError("Dataset path pattern is required")
    if args.model_repo_path == "":
        raise ValueError("Model repository path is required")
    if args.results_path == "":
        raise ValueError("Results path is required")
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    model_repo_path = args.model_repo_path
    dataset_path_pattern = args.dataset_path_pattern
    datasets_count = args.datasets_count
    results_path = args.results_path
    exp_name = args.exp_name

    collect_results(exp_name, model_repo_path, dataset_path_pattern, datasets_count, results_path)

if __name__ == '__main__':
    main()
