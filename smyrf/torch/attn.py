import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial, reduce
from itertools import chain
from smyrf.torch.utils import *


class SmyrfAttention(nn.Module):
    def __init__(self, n_hashes, q_cluster_size, k_cluster_size,
                 q_attn_size=None, k_attn_size=None,
                 clustering_algo='lsh',
                 dropout=0.,
                 # LSH clustering
                 r=1,
                 # kmeans clustering
                 max_iters=50):
        super(SmyrfAttention, self).__init__()
        self.n_hashes = n_hashes

        if q_attn_size is None:
            self.q_attn_size = q_cluster_size
        else:
            self.q_attn_size = q_attn_size

        if k_attn_size is None:
            self.k_attn_size = k_cluster_size
        else:
            self.k_attn_size = k_attn_size

        self.dropout = nn.Dropout(dropout)
        self.xbox_plus = XBOXPLUS()

        self.clustering_algo =  clustering_algo
        if clustering_algo == 'lsh':
            self.clustering_params = {
                'r': r,
                'n_hashes': self.n_hashes
            }
        else:
            raise NotImplementedError('Uknown clustering algorithm')


    def forward(self, queries, keys, values, attn_mask=None, progress=False,
                norm_factor=1):
        bs, q_seqlen, dim = queries.shape
        bs, k_seqlen, dim = keys.shape
        v_dim = values.shape[-1]
        assert queries.device == keys.device, 'Queries, keys in different devices'
        device = queries.device

        with torch.no_grad():
            # XBOX+ transform
            self.xbox_plus.set_norms(queries, keys)
            Queries = self.xbox_plus.Q(queries)
            Keys = self.xbox_plus.K(keys)

            num_clusters = Queries.shape[1] // self.q_attn_size
            assert num_clusters == (Keys.shape[1] // self.k_attn_size), 'Unequal number of clusters for queries and keys.'

            if self.clustering_algo == 'lsh':
                q_positions, k_positions = lsh_clustering(Queries, Keys, **self.clustering_params)
            else:
                raise NotImplementdError('This algorithm is not supported')

            q_positions = q_positions.reshape(self.n_hashes, bs, -1)
            k_positions = k_positions.reshape(self.n_hashes, bs, -1)

        # free memory
        del Queries
        del Keys

        q_rev_positions = torch.argsort(q_positions, dim=-1)
        q_offset = torch.arange(bs, device=queries.device).unsqueeze(-1) * q_seqlen
        k_offset = torch.arange(bs, device=queries.device).unsqueeze(-1) * k_seqlen


        q_flat = (q_positions + q_offset).reshape(-1)
        k_flat = (k_positions + k_offset).reshape(-1)

        # sorted queries, keys, values
        s_queries = queries.reshape(-1, dim).index_select(0, q_flat).reshape(-1, self.q_attn_size, dim)
        s_keys = keys.reshape(-1, dim).index_select(0, k_flat).reshape(-1, self.k_attn_size, dim)
        s_values = values.reshape(-1, v_dim).index_select(0, k_flat).reshape(-1, self.k_attn_size, v_dim)


        inner = s_queries @ s_keys.transpose(2, 1)
        inner = inner / norm_factor
        if attn_mask is not None:
            # repeat for heads (if they exist)
            attn_mask = attn_mask.unsqueeze(0).repeat(self.n_hashes, queries.shape[0] // attn_mask.shape[0], 1).reshape(-1)[k_flat].reshape(-1, self.k_attn_size)
            inner = (attn_mask.unsqueeze(1) + inner)

        # free memory
        del q_positions, k_positions

        # softmax denominator
        dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        # softmax
        dots = torch.exp(inner - dots_logsumexp)
        # dropout
        dots = self.dropout(dots)

        # n_hashes outs
        bo = (dots @ s_values).reshape(self.n_hashes, bs, q_seqlen, -1)

        # undo sort
        q_offset = torch.arange(bs * self.n_hashes, device=queries.device).unsqueeze(-1) * q_seqlen
        q_rev_flat = (q_rev_positions.reshape(-1, q_seqlen) + q_offset).reshape(-1)
        o = bo.reshape(-1, v_dim).index_select(0, q_rev_flat).reshape(self.n_hashes, bs, q_seqlen, -1)

        slogits = dots_logsumexp.reshape(self.n_hashes, bs, -1)
        logits = torch.gather(slogits, 2, q_rev_positions)

        # free memory
        del q_rev_positions

        probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
        out = torch.sum(o * probs.unsqueeze(-1), dim=0)

        return out



def dense(query, key, value):
    return F.softmax(query @ key.permute(0, 2, 1), dim=-1) @ value


if __name__ == '__main__':
    N = 1024
    dim = 30
    bs = 2
    n_hashes = 8
    q_cluster_size = k_cluster_size = 16
    device = 'cuda'

    queries = torch.randn(bs, N, dim, device=device)
    keys = torch.randn(bs, N, dim, device=device)
    values = torch.randn(bs, N, dim, device=device)

    approximator = SmyrfAttention(n_hashes, q_cluster_size, k_cluster_size)
    approximator(queries, keys, values)
