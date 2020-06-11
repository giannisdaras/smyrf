""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback
from configs import celeba_config, imagenet_config

# xla imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp
import torch_xla.debug.metrics as met
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl


def run(config):
  def len_parallelloader(self):
        return len(self._loader._loader)
  pl.PerDeviceLoader.__len__ = len_parallelloader

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  # By default, skip init if resuming training.
  if config['resume']:
    xm.master_print('Skipping initialization for training resumption...')
    config['skip_init'] = True
  config = utils.update_config_roots(config)

  # Seed RNG
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  xm.master_print('Experiment name is %s' % experiment_name)

  device = xm.xla_device(devkind='TPU')

  # Next, build the model
  G = model.Generator(**config)
  D = model.Discriminator(**config)

   # If using EMA, prepare it
  if config['ema']:
    xm.master_print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True,
                               'no_optim': True})
  else:
    xm.master_print('Not using ema...')
    G_ema, ema = None, None

  # FP16?
  if config['G_fp16']:
    xm.master_print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    xm.master_print('Casting D to fp16...')
    D = D.half()

  # Prepare state dict, which holds things like itr #
  state_dict = {'itr': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    xm.master_print('Loading weights...')
    utils.load_weights(G, D, state_dict,
                       config['weights_root'], experiment_name,
                       config['load_weights'] if config['load_weights'] else None,
                       G_ema if config['ema'] else None)

  # move everything to TPU
  G = G.to(device)
  D = D.to(device)

  G.optim = optim.Adam(params=G.parameters(), lr=G.lr,
                       betas=(G.B1, G.B2), weight_decay=0,
                       eps=G.adam_eps)
  D.optim = optim.Adam(params=D.parameters(), lr=D.lr,
                       betas=(D.B1, D.B2), weight_decay=0,
                       eps=D.adam_eps)


  #for key, val in G.optim.state.items():
  #  G.optim.state[key]['exp_avg'] = G.optim.state[key]['exp_avg'].to(device)
  #  G.optim.state[key]['exp_avg_sq'] = G.optim.state[key]['exp_avg_sq'].to(device)

  #for key, val in D.optim.state.items():
  #  D.optim.state[key]['exp_avg'] = D.optim.state[key]['exp_avg'].to(device)
  #  D.optim.state[key]['exp_avg_sq'] = D.optim.state[key]['exp_avg_sq'].to(device)


  if config['ema']:
    G_ema = G_ema.to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])

  # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D)
  xm.master_print(G)
  xm.master_print(D)
  xm.master_print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  xm.master_print('Test Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname,
                                 reinitialize=(not config['resume']))
  xm.master_print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname,
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])

  if xm.is_master_ordinal():
      # Write metadata
      utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)

  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  xm.master_print('Preparing data...')
  loader = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr']})

  # Prepare inception metrics: FID and IS
  xm.master_print('Preparing metrics...')

  get_inception_metrics = inception_utils.prepare_inception_metrics(
      config['dataset'], config['parallel'],
      no_inception=config['no_inception'],
      no_fid=config['no_fid'])

  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])

  sample = lambda: utils.prepare_z_y(G_batch_size, G.dim_z,
                                     config['n_classes'], device=device,
                                     fp16=config['G_fp16'])
  # Prepare a fixed z & y to see individual sample evolution throghout training
  fixed_z, fixed_y = sample()

  train = train_fns.GAN_training_function(G, D, GD, sample, ema, state_dict,
                                          config)

  xm.master_print('Beginning training...')

  if xm.is_master_ordinal():
    pbar = tqdm(total=config['total_steps'])
    pbar.n = state_dict['itr']
    pbar.refresh()

  xm.rendezvous('training_starts')
  while (state_dict['itr'] < config['total_steps']):
    pl_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)

    for i, (x, y) in enumerate(pl_loader):
      if xm.is_master_ordinal():
          # Increment the iteration counter
          pbar.update(1)

      state_dict['itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      G.train()
      D.train()
      if config['ema']:
        G_ema.train()

      xm.rendezvous('data_collection')
      metrics = train(x, y)

      # train_log.log(itr=int(state_dict['itr']), **metrics)

      # Every sv_log_interval, log singular values
      #if ((config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval']))) and xm.is_master_ordinal():
        #train_log.log(itr=int(state_dict['itr']),
         #             **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

      # Save weights and copies as configured at specified interval
      if (not (state_dict['itr'] % config['save_every'])):
        if config['G_eval_mode']:
          xm.master_print('Switchin G to eval mode...')
          G.eval()
          if config['ema']:
            G_ema.eval()
        train_fns.save_and_sample(G, D, G_ema, sample, fixed_z, fixed_y, state_dict, config, experiment_name)

      # Test every specified interval
      if (not (state_dict['itr'] % config['test_every'])):

        which_G = G_ema if config['ema'] and config['use_ema'] else G
        if config['G_eval_mode']:
          xm.master_print('Switchin G to eval mode...')
          which_G.eval()

        def G_sample():
            z, y = sample()
            return which_G(z, which_G.shared(y))

        train_fns.test(G, D, G_ema, sample, state_dict, config, G_sample,
                       get_inception_metrics, experiment_name, test_log)

      if state_dict['itr'] >= config['total_steps']:
          break

def main(index):
  xm.master_print(celeba_config)
  run(celeba_config)


if __name__ == '__main__':
  xmp.spawn(main, args=(), nprocs=celeba_config['num_devices'])
