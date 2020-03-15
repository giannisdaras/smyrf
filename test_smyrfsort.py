import torch
from smyrfsort import sort
from profilehooks import timecall, profile

if __name__ == '__main__':
    N = 1024 * 1024 * 20
    x = torch.LongTensor(N).random_(0, N).unsqueeze(0)

    @timecall
    def run1():
        s_y, s_ind = x.sort()

    @timecall
    def run2():
        groups = 8
        s_y, s_ind = sort(x, groups)
        s_y = s_y.reshape(groups, -1)

    run1()
    run2()
