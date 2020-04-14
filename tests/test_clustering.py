import torch
import torch.nn.functional as F
from smyrf.torch.utils import *
from smyrf.torch.attn import SmyrfAttention
import math
import logging
import pytest, os
from tqdm import tqdm
from balanced_kmeans import kmeans_equal

# Attention complexity O(N * cluster_size)
device = 'cuda'
N = 8192
dim = 30

batch_size = 1
n_hashes = 16
per_category_cluster_size = 128
num_clusters = N // per_category_cluster_size
max_iters = 100

# Input
torch.manual_seed(0)
queries = torch.randn(batch_size, N, dim, device=device)
keys = torch.randn(batch_size, N, dim, device=device)
values = torch.randn(batch_size, N, dim, device=device)

def test_clustering():
    approximator = SmyrfAttention(n_hashes, per_category_cluster_size,
                                  per_category_cluster_size,
                                  max_iters=max_iters)
    ours = approximator(queries, keys, values)
    real = F.softmax(queries @ keys.transpose(-2, -1), dim=-1) @ values
    absolute_error = torch.abs(ours - real).mean(dim=-1)
    mean_absolute_error = absolute_error.mean()
    formatted = round(mean_absolute_error.item(), 2)
    logging.log(logging.INFO, 'Mean absolute error: {}'.format(formatted))
    assert formatted <= 0.25, 'Absolute error: {}'.format(formatted)
    #

if __name__ == '__main__':
    pytest.main(args=[os.path.abspath(__file__)])
