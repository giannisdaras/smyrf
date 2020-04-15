import sys
import argparse
import torchvision
import logging
sys.path.append('biggan/')
import utils
import inception_utils
from categories import indx2category
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', type=int)
parser.add_argument('--n_hashes', type=int, default=4)
parser.add_argument('--q_cluster_size', type=int, default=32)
parser.add_argument('--q_attn_size', type=int, default=None)
parser.add_argument('--max_iters', type=int, default=50)
parser.add_argument('--imagenet_category', default='maltese')
parser.add_argument('--do_sample', default=False, action='store_true')
parser.add_argument('--do_metrics', default=False, action='store_true')
parser.add_argument('--progress', default=False, action='store_true',
                    help='Show progress of kmeans.')
parser.add_argument('--disable_smyrf', action='store_true', default=False)
# metrics configuration
parser.add_argument('--dataset', default='I128_hdf5',
                    help='Which HDF5 file to use')
parser.add_argument('--num_inception_images', default=100, type=int,
                    help='Num of images to calculate inception from')

logging.basicConfig(level=logging.INFO)

class Biggan:
    def __init__(self, config):
        self.config = config
        model = __import__(self.config["model"])
        self.generator = model.Generator(**self.config).to(self.config['device'])
        self.discriminator = model.Discriminator(**self.config).to(self.config['device'])


    def load_pretrained(self):
        state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                        'best_IS': 0, 'best_FID': 999999, 'config': self.config}
        utils.load_weights(None, self.discriminator, state_dict,
                     self.config['weights_root'],
                     self.config['experiment_name'],
                     self.config['load_weights'],
                     self.generator,
                     strict=False, load_optim=False)

        logging.log(logging.INFO, 'Weights loaded...')
        self.generator.eval()
        self.discriminator.eval()
        logging.log(logging.INFO, 'Generator and discriminator on eval mode')


    def sample(self, generator_inputs, out_path=None):
        z, y = generator_inputs
        image_tensors = self.generator(z, self.generator.shared(y))
        if out_path is not None:
            self.save_images(image_tensors.cpu(), out_path)
        return image_tensors, y


    def save_images(self, image_tensors, image_path):
        torchvision.utils.save_image(
            image_tensors,
            image_path,
            nrow=int(image_tensors.shape[0] ** 0.5),
            normalize=True,
            )

    def get_random_inputs(self, bs=1, target=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        (z_, y_) = utils.prepare_z_y(
            bs,
            self.generator.dim_z,
            self.config["n_classes"],
            device=self.config["device"],
            fp16=self.config["G_fp16"],
            z_var=self.config["z_var"],
            target=target
        )
        return (z_, y_)


if __name__ == '__main__':
    args = parser.parse_args()

    config = {
        'model': 'BigGAN',
        'G_param': 'SN',
        'D_param': 'SN',
        'G_ch': 96,
        'D_ch': 96, 'G_depth': 1,
        'D_depth': 1,
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
        'seed': 0,
        'G_init': 'ortho',
        'D_init': 'ortho',
        'skip_init': True,
        'batch_size': 1,
        'G_batch_size': 1,
        'num_G_accumulations': 8,
        'num_D_accumulations': 8,
        'num_epochs': 100,
        'G_fp16': False,
        'D_fp16': False,
        'experiment_name': '138k',
        'ema': True,
        'num_G_SVs': 1,
        'num_D_SVs': 1,
        'num_G_SV_itrs': 1,
        'num_D_SV_itrs': 1,
        'G_ortho': 0.0,
        'D_ortho': 0.0,
        'device': 'cuda',
        'n_classes': 1000,
        'load_weights': '',
        'weights_root': '/home/giannis/image2noise/image2noise/biggan_noise/',
        'experiment_name': '138k',
        'lr': 0.01,
        'optimization_steps': 600,
        # SMYRF configuration
        'smyrf': not args.disable_smyrf,
        'n_hashes': args.n_hashes,
        'q_cluster_size': args.q_cluster_size,
        'k_cluster_size': args.q_cluster_size // 4,
        'q_attn_size': args.q_attn_size,
        'k_attn_size': args.q_attn_size // 4 if args.q_attn_size else None,
        'max_iters': args.max_iters,
        'progress': args.progress,
        # Metrics configuration
        'dataset': args.dataset,
        'num_inception_images': args.num_inception_images,
    }

    biggan = Biggan(config)
    biggan.load_pretrained()

    # Random sampling
    category2indx = {val: key for key, val in indx2category.items()}
    category = category2indx[args.imagenet_category]

    if args.do_sample:
        generator_inputs = biggan.get_random_inputs(bs=args.bs,
                                                    target=category,
                                                    seed=args.seed)
        out, _ = biggan.sample(generator_inputs, out_path='./samples.png')

    if args.do_metrics:
        logging.log(logging.INFO, 'Preparing random sample sheet...')

        get_inception_metrics = inception_utils.prepare_inception_metrics(
            config['dataset'], parallel=True, no_fid=True)

        def get_metrics():

            def sample():
                images, labels = biggan.sample(biggan.get_random_inputs(bs=args.bs))
                return images, labels

            IS_mean, IS_std, FID = get_inception_metrics(
                sample, num_inception_images=config['num_inception_images'],
                num_splits=10)
            # Prepare output string
            outstring = 'Using %s weights ' % ('ema' if config['ema'] else 'non-ema')
            outstring += 'over %d images, ' % config['num_inception_images']
            outstring += 'PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (IS_mean, IS_std, FID)
            print(outstring)

        get_metrics()

    # # Sample truncation curve stuff. This is basically the same as the inception metrics code
    # if config['sample_trunc_curves']:
    #     start, step, end = [float(item) for item in config['sample_trunc_curves'].split('_')]
    #     print('Getting truncation values for variance in range (%3.3f:%3.3f:%3.3f)...' % (start, step, end))
    #     for var in np.arange(start, end + step, step):
    #       z_.var = var
    #       # Optionally comment this out if you want to run with standing stats
    #       # accumulated at one z variance setting
    #       if config['accumulate_stats']:
    #         utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
    #                                     config['num_standing_accumulations'])
    #       get_metrics()
