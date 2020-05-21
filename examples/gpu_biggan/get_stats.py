import argparse
import numpy as np


parser = argparse.ArgumentParser('Get stats from logs')
parser.add_argument('--filename', default='out.txt')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.filename, 'r') as f:
        # first five lines are useless
        lines = f.readlines()[3:]
        lines = [line.rstrip() for line in lines]
        lines = list(chunks(lines, 3))

        under_kns = []
        under0_01 = []
        maxs = []
        for a, b, c in lines:
            under_kns.append(float(a.split(' ')[-1]))
            under0_01.append(float(b.split(' ')[-1]))
            maxs.append(float(c.split(' ')[-1]))

        under_kns = np.array(under_kns)
        under0_01 = np.array(under0_01)
        maxs = np.array(maxs)

        print('Total (not considering batch size): {}'.format(len(under_kns)))
        print('Under K/N: {} +/- {}'.format(round(under_kns.mean(), 2), round(under_kns.std(), 2)))
        print('Under 0.01: {} +/- {}'.format(round(under0_01.mean(), 2), round(under0_01.std(), 2)))
        print('Max: {} +/- {}'.format(round(maxs.mean(), 2), round(maxs.std(), 2)))
