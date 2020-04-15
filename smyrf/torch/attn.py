import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial, reduce
from itertools import chain
from smyrf.torch.utils import *
from balanced_kmeans import kmeans_equal, lsh_clustering

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
            }
        elif clustering_algo == 'kmeans_equal':
            self.clustering_params = {
                'max_iters' : max_iters,
                'q_cluster_size' : q_cluster_size,
                'k_cluster_size': k_cluster_size
            }
        else:
            raise NotImplementedError('Uknown clustering algorithm')


    def forward(self, queries, keys, values, attn_mask=None, progress=False):
        bs, q_seqlen, dim = queries.shape
        bs, k_seqlen, dim = keys.shape
        v_dim = values.shape[-1]
        assert queries.device == keys.device, 'Queries, keys in different devices'
        device = queries.device

        with torch.no_grad():
            # XBOX+ transform
            self.xbox_plus.set_norms(queries, keys)
            Queries = self.xbox_plus.Q(queries).repeat(self.n_hashes, 1, 1)
            Keys = self.xbox_plus.K(keys).repeat(self.n_hashes, 1, 1)

            num_clusters = Queries.shape[1] // self.q_attn_size
            assert num_clusters == (Keys.shape[1] // self.k_attn_size), 'Unequal number of clusters for queries and keys.'

            if self.clustering_algo == 'lsh':
                q_positions = lsh_clustering(Queries, **self.clustering_params)
                k_positions = lsh_clustering(Keys, **self.clustering_params)
            elif self.clustering_algo == 'kmeans_equal':
                q_positions, k_positions = get_kmeans_buckets(Queries,
                                                              Keys,
                                                              **self.clustering_params)

            q_positions = q_positions.reshape(self.n_hashes, bs, -1)
            k_positions = k_positions.reshape(self.n_hashes, bs, -1)

        # free memory
        del Queries
        del Keys

        q_rev_positions = torch.argsort(q_positions, dim=-1)

        # sorted queries, keys, values
        s_queries = queries.unsqueeze(0).repeat(self.n_hashes, 1, 1, 1)\
                           .gather(2, q_positions.unsqueeze(-1)\
                           .repeat(1, 1, 1, dim))\
                           .reshape(-1, self.q_attn_size, dim)

        s_keys = keys.unsqueeze(0).repeat(self.n_hashes, 1, 1, 1)\
                     .gather(2, k_positions.unsqueeze(-1)\
                     .repeat(1, 1, 1, dim))\
                     .reshape(-1, self.k_attn_size, dim)

        s_values = values.unsqueeze(0).repeat(self.n_hashes, 1, 1, 1)\
                         .gather(2, k_positions.unsqueeze(-1)\
                         .repeat(1, 1, 1, v_dim))\
                         .reshape(-1, self.k_attn_size, v_dim)

        inner = s_queries @ s_keys.transpose(2, 1)
        # softmax denominator
        dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        # softmax
        dots = torch.exp(inner - dots_logsumexp).type(inner.type())
        # dropout
        dots = self.dropout(dots)

        # n_hashes outs
        bo = (dots @ s_values).reshape(self.n_hashes, bs, q_seqlen, -1)

        # undo sort
        o = bo.gather(2, q_rev_positions.unsqueeze(-1).repeat(1, 1, 1, v_dim))
        slogits = dots_logsumexp.reshape(self.n_hashes, bs, -1)
        logits = torch.gather(slogits, 2, q_rev_positions)

        probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
        out = torch.sum(o * probs.unsqueeze(-1), dim=0)
        return out


class AsymmetricLSHAttention(nn.Module):
    def __init__(self,
                 dropout=0.,
                 q_bucket_size=64,
                 k_bucket_size=64,
                 n_hashes=8,
                 add_local_attn_hash=False,
                 causal=False,
                 allow_duplicate_attention=True,
                 attend_across_buckets=True,
                 rehash_each_round=True,
                 drop_for_hash_rate=0.0,
                 return_attn=False):
        '''
            Args:
                - dropout
                - q_bucket_size
                - k_bucket_size
                - n_hashes
                - add_local_attn_hash
                - causal: If true, constraints attention only to the left side
                - allow_duplicate_attention
                - attend_across_buckets
                - rehash_each_round
                - drop_for_hash_rate
                - return_attn: Whether to return attention map
        '''
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

        self.n_hashes = n_hashes
        self.add_local_attn_hash = add_local_attn_hash

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn


    def forward(self, query, key, value, input_mask=None, input_attn_mask=None):
        batch_size, q_seqlen, dim = query.shape
        batch_size, k_seqlen, dim = key.shape
        batch_size, k_seqlen, v_dim = value.shape

        device = query.device

        # we need the same number of buckets for queries and keys
        assert (q_seqlen // self.q_bucket_size) == (k_seqlen // self.k_bucket_size)
        n_buckets = q_seqlen // self.q_bucket_size

        q_buckets = l2_hash(alsh_queries(query), n_hashes=self.n_hashes,
                            n_buckets=n_buckets, r=2.5)
        k_buckets = l2_hash(alsh_keys(key), n_hashes=self.n_hashes,
                            n_buckets=n_buckets, r=2.5)


        # (batch_size * N, n_hashes) -> (batch_size, n_hashes * N)
        q_buckets = q_buckets.reshape(batch_size, q_seqlen, self.n_hashes) \
                             .permute(0, 2, 1).reshape(batch_size, -1)

        k_buckets = k_buckets.reshape(batch_size, k_seqlen, self.n_hashes) \
                             .permute(0, 2, 1).reshape(batch_size, -1)


        if self.add_local_attn_hash:
            local_buckets = torch.full((batch_size, seqlen), n_buckets, device=device, dtype=torch.long)
            q_buckets = torch.cat((q_buckets, local_buckets), dim=1)
            k_buckets = torch.cat((k_buckets, local_buckets), dim=1)

        total_hashes = self.n_hashes + int(self.add_local_attn_hash)

        q_ticker = torch.arange(total_hashes * q_seqlen, device=device).unsqueeze(0).expand_as(q_buckets)
        k_ticker = torch.arange(total_hashes * k_seqlen, device=device).unsqueeze(0).expand_as(k_buckets)

        q_buckets_and_t = q_seqlen * q_buckets + (q_ticker % q_seqlen)
        q_buckets_and_t = q_buckets_and_t.detach()

        k_buckets_and_t = k_seqlen * k_buckets + (k_ticker % k_seqlen)
        k_buckets_and_t = k_buckets_and_t.detach()


        # Hash-based sort ("s" at the start of variable names means "sorted")
        s_q_buckets_and_t, s_q_ticker = sort_key_val(q_buckets_and_t, q_ticker, dim=-1)
        s_k_buckets_and_t, s_k_ticker = sort_key_val(k_buckets_and_t, k_ticker, dim=-1)

        _, q_undo_sort = sort_key_val(s_q_ticker, q_ticker, dim=-1)
        _, k_undo_sort = sort_key_val(s_k_ticker, k_ticker, dim=-1)
        del q_ticker
        del k_ticker

        s_q_buckets_and_t = s_q_buckets_and_t.detach()
        s_k_buckets_and_t = s_k_buckets_and_t.detach()

        s_q_ticker = s_q_ticker.detach()
        s_k_ticker = s_k_ticker.detach()

        q_undo_sort = q_undo_sort.detach()
        k_undo_sort = k_undo_sort.detach()


        q_st = (s_q_ticker % q_seqlen)
        k_st = (s_k_ticker % k_seqlen)

        sq = batched_index_select(query, q_st)
        sk = batched_index_select(key, k_st)
        sv = batched_index_select(value, k_st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets

        # reshape tickers
        bq_t = torch.reshape(q_st, (batch_size, chunk_size, -1))
        bk_t = torch.reshape(k_st, (batch_size, chunk_size, -1))

        # reshape arrays
        bq = torch.reshape(sq, (batch_size, chunk_size, -1, dim))
        bk = torch.reshape(sk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, v_dim))

        # TODO(giannisdaras): allow lookback attention

        # Dot-product attention.
        dots = torch.einsum('bhie,bhje->bhij', bq, bk)

        # TODO(giannisdaras): allow masking

        # TODO(giannidaras): allow attention between buckets

        # TODO(giannisdaras): allow duplicate attention

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type(dots.type())
        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, v_dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        class UnsortLogits(Function):
            @staticmethod
            def forward(ctx, so, slogits):
                so = so.detach()
                slogits = slogits.detach()
                o = batched_index_select(so, q_undo_sort)
                _, logits = sort_key_val(s_q_ticker, slogits, dim=-1)
                return o, logits

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                so_grad = batched_index_select(grad_x, s_q_ticker)
                _, slogits_grad = sort_key_val(q_buckets_and_t, grad_y, dim=-1)
                return so_grad, slogits_grad

        o, logits = UnsortLogits.apply(so, slogits)
        o = torch.reshape(o, (batch_size, total_hashes, q_seqlen, -1))
        logits = torch.reshape(logits, (batch_size, total_hashes, q_seqlen, 1))


        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        # TODO(giannisdaras): possibly return attention mask

        # Enable to run dense attention: out = F.softmax(query @ key.permute(0, 2, 1), dim=-1) @ value

        return out



class RandomBucketsAttention(nn.Module):
    def __init__(self,
                 q_seqlen,
                 k_seqlen,
                 dropout=0.,
                 k_bucket_size=64,
                 n_hashes=8,
                 add_local_attn_hash=False,
                 causal=False,
                 allow_duplicate_attention=True,
                 attend_across_buckets=True,
                 rehash_each_round=True,
                 drop_for_hash_rate=0.0,
                 return_attn=False,
                 max_perms=20,
                 device='cuda'):
        '''
            Args:
                - q_seqlen: query seq length
                - k_seqlen
                - dropout
                - q_bucket_size
                - k_bucket_size
                - n_hashes
                - add_local_attn_hash
                - causal: If true, constraints attention only to the left side
                - allow_duplicate_attention
                - attend_across_buckets
                - rehash_each_round
                - drop_for_hash_rate
                - return_attn: Whether to return attention map
                - max_perms: maximum batch size (needed for speed)
                - device
        '''
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.k_bucket_size = k_bucket_size

        self.n_hashes = n_hashes
        self.add_local_attn_hash = add_local_attn_hash

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # maximum number of pre-computed permutations
        self.max_perms = max_perms

        # needed for pre-computing permutations
        self.q_seqlen = q_seqlen
        self.k_seqlen = k_seqlen


        k_random_perms = torch.empty(max_perms, n_hashes, k_seqlen, dtype=torch.long, device=device)

        for i in range(max_perms):
            for j in range(n_hashes):
                k_random_perms[i][j] = torch.randperm(k_seqlen, device=device)

        hashes_offset = torch.empty(n_hashes, device=device, dtype=torch.long).fill_(k_seqlen) * torch.arange(self.n_hashes, device=device)

        # new_shape: (max_perms, k_seqlen, n_hashes)
        self.s_k_ticker = k_random_perms.permute(0, 2, 1) + hashes_offset

        del k_random_perms
        del hashes_offset

        # (max_perms, n_hashes, k_seqlen) -> (max_perms, n_hashes * k_seqlen)
        self.s_k_ticker = self.s_k_ticker.permute(0, 2, 1).reshape(max_perms, -1)

        k_ticker = torch.arange(self.n_hashes * k_seqlen, device=device).unsqueeze(0).expand_as(self.s_k_ticker)

        _, self.k_undo_sort = sort_key_val(self.s_k_ticker, k_ticker, dim=-1)

        del k_ticker


    def forward(self, query, key, value, input_mask=None, input_attn_mask=None):
        batch_size, q_seqlen, dim = query.shape
        batch_size, k_seqlen, dim = key.shape
        batch_size, k_seqlen, v_dim = value.shape


        # pre-compute more random perms than batch size
        assert batch_size <= self.max_perms

        device = query.device

        n_buckets = k_seqlen // self.k_bucket_size

        s_k_ticker = self.s_k_ticker[:batch_size].to(device)

        k_undo_sort = self.k_undo_sort[:batch_size].to(device)

        # fix range
        k_st = s_k_ticker % k_seqlen

        sq = query.repeat(self.n_hashes, 1, 1)
        sk = batched_index_select(key, k_st)
        sv = batched_index_select(value, k_st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = self.n_hashes * n_buckets

        # reshape arrays
        bq = torch.reshape(sq, (batch_size, chunk_size, -1, dim))
        bk = torch.reshape(sk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, v_dim))

        # TODO(giannisdaras): allow lookback attention

        # Dot-product attention.
        inner = torch.einsum('bhie,bhje->bhij', bq, bk)

        # TODO(giannisdaras): allow masking

        # TODO(giannidaras): allow attention between buckets

        # TODO(giannisdaras): allow duplicate attention

        # Softmax.
        dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        dots = torch.exp(inner - dots_logsumexp).type(inner.type())
        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        o = torch.reshape(bo, (batch_size, -1, v_dim))
        logits = torch.reshape(dots_logsumexp, (batch_size, -1,))
        o = torch.reshape(o, (batch_size, self.n_hashes, q_seqlen, -1))
        # ln(total mass of softmax for each query)
        logits = torch.reshape(logits, (batch_size, self.n_hashes, q_seqlen, 1))

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

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
