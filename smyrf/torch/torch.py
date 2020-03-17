import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial, reduce
from itertools import chain
from smyrfsort import sort
from pytorch_memlab import profile

def uniform(a, b, shape, device='cuda'):
    '''
        Draws shape samples from a uniform distribution U(a, b).

    '''
    return (b - a) * torch.rand(shape, device=device) + a


def sort_key_val(t1, t2, dim=-1, n_buckets=1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def max_neg_value(tensor):
    '''
        Returns -infty.
    '''
    return -torch.finfo(tensor.dtype).max



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

        q_buckets = self.l2_hash(n_buckets, self.alsh_queries(query), r=1.25)
        k_buckets = self.l2_hash(n_buckets, self.alsh_keys(key), r=200)

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


    def l2_hash(self, n_buckets, vecs, r=2.5):
        '''
            L2 Sensitive Hashing.
            Args:
                vecs: (bs, N, dim) (dtype: torch.float32)
            Output:
                buckets: (bs, n_hashes * N) (dtype: torch.int32)
        '''
        device = vecs.device
        bs, seqlen, dim = vecs.shape

        assert n_buckets % 2 == 0

        # grab attributes
        n_hashes = self.n_hashes
        alpha = torch.normal(0, 1, (dim, n_hashes), device=device)
        beta = uniform(0, r, shape=(n_hashes,), device=device)
        buckets = torch.floor(((vecs @ alpha) + beta) // r)

        # (bs, N, n_hashes) -> (n_hashes, N, bs)
        buckets = buckets.permute(2, 1, 0)
        # (n_hashes, N, bs) -> (n_hashes, N * bs)
        buckets = buckets.reshape(n_hashes, -1)

        offsets = torch.arange(n_hashes, device=device)
        offsets = torch.reshape(offsets * n_buckets, (-1, 1))

        return (buckets + offsets).reshape(-1, bs).permute(1, 0)


    def alsh_keys(self, x):
        x = x / x.max(dim=-1)[0].unsqueeze(-1)
        norm = x.norm(p=2, dim=-1).unsqueeze(-1)
        return torch.cat((x, norm**2, norm**4, norm**8), -1)

    def alsh_queries(self, x):
        # normalize queries
        x = (x - x.mean(dim=-1).unsqueeze(-1)) / x.std(dim=-1).unsqueeze(-1)
        device = x.device
        ext = torch.empty(x.shape[:-1] + (1,), device=device).fill_(0.5)
        return torch.cat((x, ext, ext, ext), -1)
