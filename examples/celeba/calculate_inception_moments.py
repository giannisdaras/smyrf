''' Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.

 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import inception_utils
from tqdm import tqdm, trange
from argparse import ArgumentParser
from configs import celeba_config

# xla imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp

def run(config):
  # Get loader
  config['drop_last'] = False
  loader = utils.get_data_loaders(**config)

  # Load inception net
  net = inception_utils.load_inception_net(parallel=config['parallel'])
  pool, logits, labels = [], [], []

  # move to TPU
  device = xm.xla_device(devkind='TPU')
  net = net.to(device)

  for i, (x, y) in enumerate(tqdm(loader)):
    x = x.to(device)
    with torch.no_grad():
      pool_val, logits_val = net(x)
      pool += [np.asarray(pool_val.cpu())]
      logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
      labels += [np.asarray(y.cpu())]

  pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
  # uncomment to save pool, logits, and labels to disk
  print('Saving pool, logits, and labels to disk...')
  np.savez(config['dataset']+'_inception_activations.npz',
            {'pool': pool, 'logits': logits, 'labels': labels})
  # Calculate inception metrics and report them
  print('Calculating inception metrics...')
  IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
  print('Training data from dataset %s has IS of %5.5f +/- %5.5f' % (config['dataset'], IS_mean, IS_std))
  # Prepare mu and sigma, save to disk. Remove "hdf5" by default
  # (the FID code also knows to strip "hdf5")
  print('Calculating means and covariances...')
  mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
  print('Saving calculated means and covariances to disk...')
  np.savez(config['dataset'].strip('_hdf5')+'_inception_moments.npz', **{'mu' : mu, 'sigma' : sigma})


if __name__ == '__main__':
  config = celeba_config
  print(config)
  run(config)
