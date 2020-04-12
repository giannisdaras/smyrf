import torch
import torch.nn.functional as F
from smyrf.torch.utils import *
import math
import logging
import sys
import pytest, os
from knn_cuda import KNN
from profilehooks import timecall
from tqdm import tqdm
import random
from kmeans_pytorch import kmeans_equal

# Attention complexity O(N * cluster_size)
device = 'cuda'
N = 8192
dim = 30

n_hashes = 32
per_category_cluster_size = 512
num_clusters = N // per_category_cluster_size


# Input
torch.manual_seed(0)
queries = torch.randn(N, dim, device=device)
keys = torch.randn(N, dim, device=device)
values = torch.randn(N, dim, device=device)

# Transformations
xbox = XBOX()

print('Loading kNN')
knn = KNN(k=per_category_cluster_size, transpose_mode=True)


def get_competitive_matching_buckets(Queries, Keys):
    # IMPORTANT: for repetitive runs, consider permuting the sequences first
    # permute sequence
    # perm_ticker = torch.randperm(N)
    # perm_keys = keys[perm_ticker]
    # perm_queries = queries[perm_ticker]
    # perm_Keys = Keys[perm_ticker]
    # perm_Queries = Queries[perm_ticker]
    # q_positions[i] = perm_ticker[q_indices]
    # k_positions[i] = perm_ticker[k_indices]

    index = 0
    q_buckets = []
    k_buckets = []
    remaining_queries = set([x for x in range(N)])
    # find for keys, using as "knowledge" queries.
    k_q_indices = knn(Queries.unsqueeze(0), Keys.unsqueeze(0))[1][0]
    # find for queries, using as "knowledge" keys.
    q_k_indices = knn(Keys.unsqueeze(0), Queries.unsqueeze(0))[1][0]
    for i in tqdm(range(N)):
        if len(q_buckets) == N: break
        if i in remaining_queries:
            remaining_queries.remove(i)
            found_keys = [x.item() for x in q_k_indices[i]]

            found_queries = [i]
            for k_ind in found_keys:
                matched_queries = [x.item() for x in k_q_indices[k_ind]]
                for q_ind in matched_queries:
                    if q_ind in remaining_queries:
                        found_queries.append(q_ind)
                        remaining_queries.remove(q_ind)
                    if len(found_queries) == per_category_cluster_size:
                        break

                if len(found_queries) == per_category_cluster_size:
                    break

            while len(found_queries) < per_category_cluster_size:
                q_ind = random.sample(remaining_queries, 1)[0]
                remaining_queries.remove(q_ind)
                found_queries.append(q_ind)

            q_buckets.extend(found_queries)
            k_buckets.extend(found_keys)
    q_indices = torch.tensor(q_buckets, device=device)
    k_indices = torch.tensor(k_buckets, device=device)
    return q_indices, k_indices


def get_kmeans_buckets(Queries, Keys):
    num_clusters = Queries.shape[0] // per_category_cluster_size
    q_buckets = kmeans_equal(Queries.unsqueeze(0), num_clusters=num_clusters,
                             cluster_size=per_category_cluster_size)[0]
    k_buckets = kmeans_equal(Keys.unsqueeze(0), num_clusters=num_clusters,
                             cluster_size=per_category_cluster_size)[0]
    return q_buckets.argsort(dim=-1), k_buckets.argsort(dim=-1)


def test_clustering():
    q_positions = torch.empty(n_hashes, N, dtype=torch.long, device=device)
    k_positions = torch.empty(n_hashes, N, dtype=torch.long, device=device)

    Queries = xbox.Q(queries)
    Keys = xbox.K(keys)


    for i in tqdm(range(n_hashes)):
        # get buckets
        q_indices, k_indices = get_kmeans_buckets(Queries, Keys)
        q_positions[i] = q_indices
        k_positions[i] = k_indices

    q_rev_positions = torch.argsort(q_positions, dim=-1)
    s_queries = queries[q_positions].reshape(n_hashes * num_clusters, per_category_cluster_size, dim)
    s_keys = keys[k_positions].reshape(n_hashes * num_clusters, per_category_cluster_size, dim)
    s_values = values[k_positions].reshape(n_hashes * num_clusters, per_category_cluster_size, dim)


    inner = s_queries @ s_keys.transpose(2, 1)
    # softmax denominator
    dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
    # softmax
    dots = torch.exp(inner - dots_logsumexp).type(inner.type())

    # n_hashes outs
    bo = (dots @ s_values).reshape(n_hashes, N, -1)

    # permutation
    o = batched_index_select(bo, q_rev_positions)

    slogits = dots_logsumexp.reshape(n_hashes, -1)
    # permutation
    logits = torch.gather(slogits, 1, q_rev_positions)

    probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
    ours = torch.sum(o * probs.unsqueeze(-1), dim=0)


    real = F.softmax(queries @ keys.transpose(1, 0), dim=-1) @ values
    absolute_error = torch.abs(ours - real).mean(dim=-1)
    mean_absolute_error = absolute_error.mean()
    import pdb; pdb.set_trace()
    formatted = round(mean_absolute_error.item(), 2)
    logging.log(logging.INFO, 'Mean absolute error: {}'.format(formatted))
    assert formatted <= 0.1, 'Absolute error: {}'.format(formatted)


if __name__ == '__main__':
    pytest.main(args=[os.path.abspath(__file__)])
