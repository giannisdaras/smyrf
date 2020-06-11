from main import Biggan
from configs import config
import argparse
from smyrf.torch.utils import color_clusters
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
# Configure matplotlib with Latex font
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import random
import numpy as np
from categories import indx2category
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--weights_root', default='.')
parser.add_argument('--experiment_name', default='130k')
parser.add_argument('--ema', default=False, action="store_true")
parser.add_argument('--device', default='cuda')
# Good seeds: {200}
parser.add_argument('--seed', type=int)
parser.add_argument('--imagenet_category', default=None)
parser.add_argument('--bs', type=int, default=6)

if __name__ == '__main__':
    args = parser.parse_args()
    # parameterize config
    config['experiment_name'] = args.experiment_name
    config['ema'] = args.ema
    config['weights_root'] = args.weights_root
    config['smyrf'] = False
    config['return_attn_map'] = True

    biggan = Biggan(config)
    biggan.load_pretrained()

    # Random sampling
    category2indx = {val: key for key, val in indx2category.items()}
    if args.imagenet_category is not None:
        category = category2indx[args.imagenet_category]
    else:
        category = None

    generator_inputs = biggan.get_random_inputs(bs=args.bs,
                                                target=category,
                                                seed=args.seed)
    out, attn_map, _ = biggan.sample(generator_inputs, return_attn_map=True)

    print('Computing singular values...')

    fig = plt.figure()
    fig.suptitle(f'Singular values of BigGAN\'s attention map (after softmax) for {args.bs} randomly generated images.')
    plt.xlabel('Index')
    plt.ylabel('Singular value')
    for i in tqdm(range(args.bs)):
        u, s, vh = np.linalg.svd(attn_map[i].squeeze().cpu().detach().numpy())
        plt.plot(np.arange(len(s)), s)

    fig.savefig('../../visuals/singular_values.png')
    plt.show()
