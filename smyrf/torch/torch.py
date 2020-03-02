import torch
import torch.nn as nn
import math
from profilehooks import timecall, profile, coverage

def handle_device(device):
    if device == 'tpu':
        import torch_xla_py.xla_model as xm
        device = xm.xla_device()
    return device


def make_unit_length(x, epsilon=1e-6):
    variance = torch.mean(x**2, dim=-1, keepdim=True)
    norm_inputs = x / torch.sqrt(variance + epsilon)
    return norm_inputs


def uniform(a, b, shape, device='cuda'):
    return (b - a) * torch.rand(shape, device=device) + a


class ApproximateAttention(nn.Module):
    def __init__(self, q_dim, k_dim, n_buckets, n_hashes, n_bins,
                 r=2.5, hash_method='alsh', device='cuda'):
        '''
            Args:
                - n_hashes:

                - n_buckets: This parameter controls the number of buckets. Nodes
                get hashed in buckets and then splitted to bins based on their
                bucket index. Generally, having a lot of buckets leads to better
                seperation, but is computationally intensive.

                - n_bins: This parameter controls the number of separate attentions.
                Nodes are separated in bins and nodes in a bin attend only within
                the bin. If you increase this parameter, you should expect
                acceleration but also worse results. With n_bins=1,
                this is equivalent to dense attention.


                - hash_method:
                    Default: alsh.
        '''
        super(ApproximateAttention, self).__init__()
        self.n_buckets = n_buckets
        self.n_hashes = n_hashes
        self.n_bins = n_bins
        self.hash_method = hash_method
        self.device = device
        self.q_dim = q_dim
        self.k_dim = k_dim

        # for L2 hash
        self.alpha = torch.normal(0, 1, (q_dim + 3, n_hashes * n_buckets),
                                  device=device)
        self.r = 2.5
        self.beta = uniform(0, r, shape=(n_hashes * n_buckets), device=device)
        self.bucket_weights = torch.normal(0, 1, (n_buckets,), device=device)

        # save these variables for efficient repetitive calls.
        self.bs = None
        self.q_ticker = None
        self.k_ticker = None
        self.q_seqlen = None
        self.k_seqlen = None
        self.offsets = None


    def forward(self, query, key, value):
        device = query.device
        # lengths
        q_seqlen = query.shape[-2]
        k_seqlen = key.shape[-2]
        bs = query.shape[0]

        # embeddings
        q_E = query.shape[-1]
        k_E = key.shape[-1]
        v_E = value.shape[-1]

        # class attributes (for shorter code)
        n_hashes = self.n_hashes
        n_buckets = self.n_buckets
        n_bins = self.n_bins

        with torch.no_grad():
            if self.hash_method == 'alsh':
                q_ext = self.alsh_queries(query)
                k_ext = self.alsh_keys(key)
                # q_buckets: (bs, n_hashes * q_seqlen)
                q_buckets = self.l2_hash(q_ext)
                k_buckets = self.l2_hash(k_ext)
            else:
                raise ValueError('Only alsh supported.')

            # ------------------ sort based on their buckets -------------------- #
            if q_seqlen != self.q_seqlen or (bs != self.bs):
                self.q_ticker = torch.arange(n_hashes * q_seqlen, device=device).unsqueeze(0).expand(bs, n_hashes * q_seqlen)

            if k_seqlen != self.k_seqlen or (bs != self.bs):
                self.k_ticker = torch.arange(n_hashes * k_seqlen, device=device).unsqueeze(0).expand(bs, n_hashes * k_seqlen)

            # retrieve
            k_ticker = self.k_ticker
            q_ticker = self.q_ticker

            q_buckets_t = q_seqlen * q_buckets + (q_ticker % q_seqlen)
            k_buckets_t = k_seqlen * k_buckets + (k_ticker % k_seqlen)

            _, s_q_ticker = torch.sort(q_buckets_t, dim=-1)
            _, s_k_ticker = torch.sort(k_buckets_t, dim=-1)

            _, q_undo_sort = torch.sort(s_q_ticker, dim=-1)
            _, k_undo_sort = torch.sort(s_k_ticker, dim=-1)


            if self.bs != bs or q_seqlen != self.q_seqlen:
                self.query_offset = (torch.arange(0, bs, device=device) * q_seqlen).unsqueeze(1)
                self.query_undo_offset = (torch.arange(0, bs, device=device) * q_seqlen * n_hashes).unsqueeze(1)

            if self.bs != bs or k_seqlen != self.k_seqlen:
                self.key_offset = (torch.arange(0, bs, device=device) * k_seqlen).unsqueeze(1)
                self.key_undo_offset = (torch.arange(0, bs, device=device) * k_seqlen * n_hashes).unsqueeze(1)

            # save for next call
            self.q_seqlen = q_seqlen
            self.k_seqlen = k_seqlen
            self.bs = bs

            # get self variables
            query_offset = self.query_offset
            query_undo_offset = self.query_undo_offset
            key_offset = self.key_offset
            key_undo_offset = self.key_undo_offset

            q_t = (s_q_ticker % q_seqlen) + query_offset
            k_t = (s_k_ticker % k_seqlen) + key_offset

            q_undo_sort = q_undo_sort + query_undo_offset
            k_undo_sort = k_undo_sort + key_undo_offset

        s_q = query.reshape(-1, q_E)[q_t]
        s_k = key.reshape(-1, k_E)[k_t]
        s_v = value.reshape(-1, v_E)[k_t]

        # ----------------- split to bins  -------------------------------------- #
        b_q = torch.reshape(s_q, (bs, n_hashes * n_bins, -1, s_q.shape[-1]))
        b_k = torch.reshape(s_k, (bs, n_hashes * n_bins, -1, s_k.shape[-1]))
        b_v = torch.reshape(s_v, (bs, n_hashes * n_bins, -1, s_v.shape[-1]))

        # -------------- calculate inside bins attention ------------------------- #

        # bs, n_hashes * n_bins, per_bin_a, per_bin_b <- (bs, n_hashes * n_bins, per_bin_a, dim) * (bs, n_hashes * n_bins, dim, per_bin_b)
        dots = torch.einsum('bkxd, bkyd -> bkxy', b_q, b_k) # eq. to: torch.matmul(b_q, torch.transpose(b_k, -1, -2))

        # softmax
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp)

        # ----------------- project with values ---------------------------------- #
        # bs, n_hashes * n_bins, per_bin_a, E
        bo = torch.matmul(dots, b_v)

        # undo sort
        o = bo.reshape(-1, bo.shape[-1])[q_undo_sort]
        logits = dots_logsumexp.reshape(-1, 1)[q_undo_sort]

        o = torch.reshape(o, (bs, n_hashes, q_seqlen, -1))
        logits = torch.reshape(logits, (bs, n_hashes, q_seqlen, -1))

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)
        return out


    def l2_hash(self, vecs):
        '''
            L2 Sensitive Hashing.
            Args:
                vecs: (bs, N, dim) (dtype: torch.float32)
            Output:
                buckets: (bs, n_hashes * N) (dtype: torch.int32)
        '''
        device = vecs.device
        bs = vecs.shape[0]

        # grab attributes
        n_hashes = self.n_hashes
        n_buckets = self.n_buckets
        r = self.r
        alpha = self.alpha
        beta = self.beta
        bucket_weights = self.bucket_weights

        hashed_vecs = ((vecs @ alpha) + beta) // r
        buckets = torch.matmul(hashed_vecs.reshape(bs, -1, n_hashes, n_buckets), bucket_weights).type(torch.int32) % n_buckets

        # (bs, N, n_hashes) -> (n_hashes, N, bs)
        buckets = buckets.permute(2, 1, 0)
        # (n_hashes, N, bs) -> (n_hashes, N * bs)
        buckets = buckets.reshape(n_hashes, -1)

        # offset different hashes
        if self.offsets is None:
            self.offsets = torch.arange(n_hashes, device=device)
            self.offsets = torch.reshape(self.offsets * n_buckets, (-1, 1))

        return (buckets + self.offsets).reshape(-1, bs).permute(1, 0)


    def alsh_keys(self, x):
        norm = x.norm(p=2, dim=-1).unsqueeze(-1)
        return torch.cat((x, norm**2, norm**4, norm**8), -1)

    def alsh_queries(self, x):
        device = x.device
        ext = torch.empty(x.shape[:-1] + (1,), device=device).fill_(0.5)
        return torch.cat((x / x.norm(p=2, dim=-1).unsqueeze(-1), ext, ext, ext), -1)
