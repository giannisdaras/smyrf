import torch
import math
from profilehooks import timecall


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


def angular_hash(vecs, n_buckets=64, n_hashes=1):
    '''
        Args:
            vecs: Numpy array. Shape: (N, E)

        Returns: Numpy array. Shape: (N, n_hashes)
    '''

    device = vecs.device
    rot_size, factor_list = n_buckets, [n_buckets]

    rotations_shape = (
        vecs.shape[-1],
        n_hashes,
        rot_size // 2)

    random_rotations = uniform(0, 1, rotations_shape)

    rotated_vecs = torch.einsum('tf,fhb->htb', vecs, random_rotations)
    rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)

    buckets = torch.argmax(rotated_vecs, axis=-1)

    # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
    # bucket numbers from different hashing rounds don't overlap.
    offsets = torch.arange(n_hashes, device=device)
    offsets = torch.reshape(offsets * n_buckets, (-1, 1))
    buckets = torch.reshape(buckets + offsets, (-1,))
    return buckets

def l2_hash(vecs, n_buckets=64, n_hashes=1, alpha=None, beta=None, r=2.5):
    device = vecs.device
    bs = vecs.shape[0]

    if alpha is None:
        alpha = torch.normal(0, 1, (vecs.shape[-1], n_hashes * n_buckets), device=device)
    if beta is None:
        beta = uniform(0, r, shape=(n_hashes * n_buckets), device=device)

    hashed_vecs = ((vecs @ alpha) + beta) // r
    buckets = torch.matmul(hashed_vecs.reshape(bs, -1, n_hashes, n_buckets), torch.normal(0, 1, (n_buckets,), device=device)) % n_buckets
    # (bs, N, n_hashes) -> (n_hashes, N, bs)
    buckets = buckets.permute(2, 1, 0)
    # (n_hashes, N, bs) -> (n_hashes, N * bs)
    buckets = buckets.reshape(n_hashes, -1)

    # offset different hashes
    offsets = torch.arange(n_hashes, device=device)
    offsets = torch.reshape(offsets * n_buckets, (-1, 1))
    # (n_hashes, N, bs) -> (n_hashes * N, bs)
    return (buckets + offsets).reshape(-1, bs).permute(1, 0)


def alsh_keys(x):
    norm = x.norm(p=2, dim=-1).unsqueeze(-1)
    return torch.cat((x, norm**2, norm**4, norm**8), -1)

def alsh_queries(x):
    device = x.device
    ext = torch.empty(x.shape[:-1] + (1,), device=device).fill_(0.5)
    return torch.cat((x / x.norm(p=2, dim=-1).unsqueeze(-1), ext, ext, ext), -1)


def normal_attention(q, k, v):
    softmaxed = torch.nn.functional.softmax(q @ k.transpose(-1, 0))
    out = softmaxed @ v
    return out


def estimate_attention(query, key, value, n_buckets=64, n_hashes=1, n_bins=5,
                       hash_method='angular_hash'):
    device = query.device

    # lengths
    q_seqlen = query.shape[-2]
    k_seqlen = key.shape[-2]

    # embeddings
    q_E = query.shape[-1]
    k_E = key.shape[-1]
    v_E = value.shape[-1]

    bs = query.shape[0]

    if hash_method == 'alsh':
        q_ext = alsh_queries(query)
        k_ext = alsh_keys(key)
        q_buckets = l2_hash(q_ext, n_buckets=n_buckets, n_hashes=n_hashes)
        k_buckets = l2_hash(k_ext, n_buckets=n_buckets, n_hashes=n_hashes)

    # ------------------ sort based on their buckets ----------------------- #

    # enumerate them
    q_ticker = torch.arange(n_hashes * q_seqlen, device=device).unsqueeze(0).expand(bs, n_hashes * q_seqlen)
    k_ticker = torch.arange(n_hashes * k_seqlen, device=device).unsqueeze(0).expand(bs, n_hashes * k_seqlen)

    q_buckets_t = q_seqlen * q_buckets + (q_ticker % q_seqlen)
    k_buckets_t = k_seqlen * k_buckets + (k_ticker % k_seqlen)

    # sort / unsort
    query_offset = (torch.arange(0, bs, device=device) * q_seqlen).unsqueeze(1)
    key_offset = (torch.arange(0, bs, device=device) * k_seqlen).unsqueeze(1)

    query_undo_offset = (torch.arange(0, bs, device=device) * q_seqlen * n_hashes).unsqueeze(1)
    key_undo_offset = (torch.arange(0, bs, device=device) * k_seqlen * n_hashes).unsqueeze(1)

    _, s_q_ticker = torch.sort(q_buckets_t, dim=1)
    _, s_k_ticker = torch.sort(k_buckets_t, dim=1)

    _, q_undo_sort = torch.sort(s_q_ticker, dim=1)
    _, k_undo_sort = torch.sort(s_k_ticker, dim=1)

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

    # bs, n_hashes * n_bins, per_bin_a, per_bin_b
    dots = torch.matmul(b_q, torch.transpose(b_k, -1, -2))
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
