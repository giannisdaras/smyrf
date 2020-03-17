import torch
from smyrfsort import w_sort as smyrfsort

def test_indices():
    x = torch.arange(100).unsqueeze(0)
    num_buckets = 10
    s_x, ind = smyrfsort(x, num_buckets)
    print(ind)
    assert ind == x


if __name__ == '__main__':
    test_indices()
