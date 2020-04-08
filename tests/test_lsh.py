import torch
from smyrf.torch.utils import *
import math
import logging
import sys
import pytest,os

def get_suggested_L_K(N):
    '''
        Constant c heavily depends on vector dimension.

    '''
    L = math.sqrt(N)
    K = math.log(N) / math.log(2)
    return int(L), int(K)



L = 128
K = 64
device = 'cuda'
N = 4096
dim = 30
check_threshold = 64

if L is None or K is None:
    L, K = get_suggested_L_K(N)

torch.manual_seed(0)
query = torch.randn(1, dim, device=device)
keys = torch.randn(N, dim, device=device)


# NN schemes
voronoi = VoronoiLSH(L=L, K=K, dim=dim, device=device)
voronoi1 = VoronoiLSH(L=L, K=K, dim=dim + 1, device=device)
voronoi3 = VoronoiLSH(L=L, K=K, dim=dim + 3, device=device)

crosspolytope = CrossPolytopeLSH(L=L, K=K, dim=dim, device=device)
crosspolytope1 = CrossPolytopeLSH(L=L, K=K, dim=dim + 1, device=device)
crosspolytope3 = CrossPolytopeLSH(L=L, K=K, dim=dim + 3, device=device)

e2lsh = E2LSH(n_hashes=L * K, dim=dim)
qlsh = QLSH(n_hashes=L * K, dim=dim)
qlsh1 = QLSH(n_hashes=L * K, dim=dim + 1)

# Transformations
_xbox = XBOX()
_h2lsh = H2LSH()
_l2lsh = L2LSH()


def check_validity(hash_agreement, gold, descending, header=''):
    s_agreements_ind = torch.argsort(hash_agreement, descending=True)
    s_gold_ind = torch.argsort(gold, descending=descending)
    seqa = [x.item() for x in s_agreements_ind[:check_threshold]]
    seqb = [x.item() for x in s_gold_ind[:check_threshold]]

    # get stats
    count, not_found = inversion_number(seqa, seqb)
    average_error = round(count / check_threshold, 3)
    inversion_info = ' Average error in relative positions: {} Not found in gold: {}'.format(average_error, not_found)
    logging.log(logging.INFO, inversion_info)

    msg = header + ' More than half options are wrong.'.format(check_threshold)
    thres = len(seqa) // 2
    assert not_found < thres, msg


def test_voronoi(caplog):
    q_hash = voronoi(query)
    keys_hash = voronoi(keys)
    hash_agreement = (q_hash == keys_hash).sum(dim=-1)

    norms = torch.norm(query - keys, p=2, dim=-1)
    check_validity(hash_agreement, norms, descending=False,
                   header='[Voronoi-NN]')

def test_e2lsh():
    q_hash = e2lsh(query)
    keys_hash = e2lsh(keys)
    hash_agreement = (q_hash == keys_hash).sum(dim=-1)
    norms = torch.norm(query - keys, p=2, dim=-1)
    check_validity(hash_agreement, norms, descending=False,
                   header='[E2LSH-NN]')


def test_qlsh():
    hash_agreement = qlsh(query, keys)
    norms = torch.norm(query - keys, p=2, dim=-1)
    check_validity(hash_agreement, norms, descending=False,
                   header='[QLSH-NN]')


def test_l2lsh_voronoi():
    Query = _l2lsh.Q(query)
    Keys = _l2lsh.K(keys)
    q_hash = voronoi3(Query)
    keys_hash = voronoi3(Keys)
    hash_agreement = (q_hash == keys_hash).sum(dim=-1)
    inner = (query @ keys.transpose(-1, -2))[0]
    check_validity(hash_agreement, inner, descending=True,
                   header='[Voronoi-MIPS-L2LSH]')

def test_xbox_voronoi():
    Query = _xbox.Q(query)
    Keys = _xbox.K(keys)
    q_hash = voronoi1(Query)
    keys_hash = voronoi1(Keys)
    hash_agreement = (q_hash == keys_hash).sum(dim=-1)
    inner = (query @ keys.transpose(-1, -2))[0]
    check_validity(hash_agreement, inner, descending=True,
                   header='[Voronoi-MIPS-XBOX]')

def test_xbox_crosspolytope():
    Query = _xbox.Q(query)
    Keys = _xbox.K(keys)
    q_hash = crosspolytope1(Query)
    keys_hash = crosspolytope1(Keys)
    hash_agreement = (q_hash == keys_hash).min(dim=-1)[0].sum(dim=-1)
    inner = (query @ keys.transpose(-1, -2))[0]
    check_validity(hash_agreement, inner, descending=True,
                   header='[CROSSPOLYTOPE-MIPS-XBOX]')


def test_h2lsh_voronoi():
    Query = _h2lsh.K(keys)
    Keys = _h2lsh.Q(query)
    q_hash = voronoi1(Query)
    keys_hash = voronoi1(Keys)
    hash_agreement = (q_hash == keys_hash).sum(dim=-1)
    inner = (query @ keys.transpose(-1, -2))[0]
    check_validity(hash_agreement, inner, descending=True,
                   header='[Voronoi-MIPS-H2LSH]')

def test_xbox_qlsh():
    Query = _xbox.Q(query)
    Keys = _xbox.K(keys)
    hash_agreement = qlsh1(Query, Keys)
    inner = (query @ keys.transpose(-1, -2))[0]
    check_validity(hash_agreement, inner, descending=True,
                   header='[QLSH-MIPS-XBOX]')

def test_h2lsh_qlsh():
    Query = _h2lsh.K(keys)
    Keys = _h2lsh.Q(query)
    hash_agreement = qlsh1(Query, Keys)
    inner = (query @ keys.transpose(-1, -2))[0]
    check_validity(hash_agreement, inner, descending=True,
                   header='[QLSH-MIPS-H2LSH]')

if __name__ == '__main__':
    pytest.main(args=[os.path.abspath(__file__)])
