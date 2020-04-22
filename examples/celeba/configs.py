celeba_config = {
    ## Dataset config
    'dataset': 'celeba',
    'augment': False,
    'num_workers': 8,
    'no_pin_memory': True,
    'shuffle': True,
    'load_in_mem': False,
    'use_multiepoch_sampler': False,

    ## Model
    'model': 'BigGAN',
    'G_param': 'SN',
    'D_param': 'SN',
    'G_ch': 96,
    'D_ch': 96,
    'G_depth': 1,
    'D_depth': 1,
    'D_thin': True,
    'D_wide': True,
    'G_shared': True,
    'shared_dim': 128,
    'dim_z': 120,
    'z_var': 1.0,
    'hier': True,
    'cross_replica': False,
    'G_nl': 'inplace_relu',
    'D_nl': 'inplace_relu',
    'G_attn': '64',
    'D_attn': '64',
    'mybn': False,
    'norm_style': 'bn',

    ## Initialization
    'seed': 0,
    'G_init': 'ortho',
    'D_init': 'ortho',
    'skip_init': True,

    ## Optimization
    'batch_size': 64,
    'G_batch_size': 0,
    'num_G_accumulations': 1,
    'num_D_accumulations': 1,
    'num_D_steps': 1,
    'G_lr': 1e-4,
    'D_lr': 4e-4,
    'G_B1': 0.,
    'D_B1': 0.,
    'G_B2': 0.999,
    'D_B2': 0.999,
    'num_epochs': 500,

    ## Hardware specs
    'device': 'cpu', # TPU POD will overwrite this
    'num_devices': 8, # i.e a TPUv3-8 has 8 devices
    'parallel': 'False',

    ## Others:
    'split_D': False,
    'config_from_name': False,
    'G_eval_mode': False,
    'D_eval_mode': False,

    # Precision
    'G_fp16': False,
    'D_fp16': False,
    'D_mixed_precision': False,
    'G_mixed_precision': False,
    'accumulate_stats': False,
    'num_standing_accumulations': 16,


    ## EMA configuration
    'ema': True,
    'use_ema': True,
    'ema_decay': 0.9999,
    'ema_start': 20000,
    ## Numerical and SV stuff
    'adam_eps': 1e-6,
    'BN_eps': 1e-5,
    'SN_eps': 1e-6,
    'num_G_SVs': 1,
    'num_D_SVs': 1,
    'num_G_SV_itrs': 1,
    'num_D_SV_itrs': 1,
    ## Ortho stuff
    'G_ortho': 0.0,
    'D_ortho': 0.0,
    'toggle_grads': False,

    ## Training configuration
    'which_train_fn': 'GAN',

    ## Logging
    'logstyle': '%3.3e',
    'log_G_spectra': False,
    'log_D_spectra': False,
    'sv_log_interval': 10,
    'pbar': 'mine',
    'name_suffix': '',

    'optimization_steps': 600,
    # SMYRF configuration
    'smyrf': False,
    'clustering_algo': 'lsh',
    'n_hashes': 8,
    'q_cluster_size': 1024,
    'k_cluster_size': 256,
    'q_attn_size': 1024,
    'k_attn_size': 256,
    ## K-means
    'max_iters': 30,
    'progress': False,
    ## LSH
    'r': 4,
    # Checkpointing and testing
    'num_inception_images': 10000,
    'test_every': 5000,
    'save_every': 2000,
    'num_save_copies': 2,
    'num_best_copies': 5,
    'which_best': 'FID',
    'no_fid': False,
    'no_inception': True, # there is no meaning in getting Inception score in Celeba
    'resume': False,
    'load_weights': '',
    'base_root': '',
    'weights_root': 'celeba_weights',
    'data_root': 'CelebAMask-HQ/CelebA-HQ-img',
    'logs_root': 'logs/',
    'samples_root': 'samples/',
    'experiment_name': 'celeba1024',
    'hash_name': False,
}
