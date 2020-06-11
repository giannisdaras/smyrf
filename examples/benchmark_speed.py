from smyrf.torch.attn import SmyrfAttention
import argparse
from time import time
from tqdm import tqdm
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn.functional as F

def measure_time(fn):
    start_time = time()
    fn()
    end_time = time()
    return end_time - start_time


def prepare_input(batch_size, seq_length, q_dim, v_dim, device='cuda'):
    queries = torch.randn(batch_size, seq_length, q_dim, device=device)
    keys = torch.randn(batch_size, seq_length, q_dim, device=device)
    values = torch.randn(batch_size, seq_length, v_dim, device=device)
    return queries, keys, values


HASHES = [1, 2, 4, 8]
BATCHES = [32, 16, 8, 4, 2, 1]

parser = argparse.ArgumentParser('Parser for speed benchmarks')
parser.add_argument('--q_dim', type=int, default=10)
parser.add_argument('--v_dim', type=int, default=10)
parser.add_argument('--seq_length', type=int, default=16184)
parser.add_argument('--trials', type=int, default=50)
parser.add_argument('--device', default='cuda')
parser.add_argument('--cluster_size', type=int, default=4)


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.trials >= 5, 'Use at least 5 trials...'
    times = defaultdict(list)
    x_axis = defaultdict(list)

    for n_hashes in tqdm(HASHES, 'Running hashes...'):
        smyrf = SmyrfAttention(n_hashes, args.cluster_size, args.cluster_size)

        for batch in tqdm(BATCHES, 'Running batches...'):
            batch_vals = []
            seq_length = args.seq_length // batch
            queries, keys, values = prepare_input(batch, seq_length, args.q_dim, args.v_dim, device=args.device)
            lambda_fn = lambda: smyrf(queries, keys, values)

            # sampling
            for i in tqdm(range(args.trials), 'Running trials...'):
                batch_vals.append(measure_time(lambda_fn))

            # remove first 5 because are noisy
            batch_val = sum(batch_vals[5:]) / len(batch_vals)

            times['smyrf_hashes{}'.format(n_hashes)].append(batch_val)
            x_axis['smyrf_hashes{}'.format(n_hashes)].append(seq_length)

    print('\n================= SMYRF OVER =============================\n')
    for batch in tqdm(BATCHES, 'Running batches...'):
        batch_vals = []
        seq_length = args.seq_length // batch
        queries, keys, values = prepare_input(batch, seq_length, args.q_dim, args.v_dim, device=args.device)
        lambda_fn = lambda: F.softmax(queries @ keys.transpose(-2, -1), dim=-1) @ values

        # sampling
        for i in tqdm(range(args.trials), 'Running trials...'):
            batch_vals.append(measure_time(lambda_fn))

        # remove first 5 because are noisy
        batch_val = sum(batch_vals[5:]) / len(batch_vals)

        times['dense'].append(batch_val)
        x_axis['dense'].append(seq_length)

    print('\n================= Dense OVER =============================\n')

    fig = plt.figure()
    for identifier in times.keys():
        plt.plot(x_axis[identifier], times[identifier])

    plt.legend([x for x in times.keys()])
    fig.savefig('../visuals/speed.png')

    plt.show()
