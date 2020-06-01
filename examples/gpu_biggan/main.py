import sys
import argparse
import torchvision
import logging
import utils
import inception_utils
from categories import indx2category
import torch
from tqdm import tqdm, trange
import numpy as np
from configs import config


parser = argparse.ArgumentParser()
parser.add_argument('--weights_root', default='.')
parser.add_argument('--experiment_name', default='130k')
parser.add_argument('--ema', default=False, action="store_true")
parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', type=int)
# Clustering configuration
parser.add_argument('--n_hashes', type=int, default=4)
parser.add_argument('--q_cluster_size', type=int, default=32)

## LSH
parser.add_argument('--r', default=1, type=float)

parser.add_argument('--imagenet_category', default=None)
parser.add_argument('--do_sample', default=False, action='store_true')
parser.add_argument('--sample_iters', default=1, type=int)
parser.add_argument('--do_metrics', default=False, action='store_true')
parser.add_argument('--disable_smyrf', action='store_true', default=False)
parser.add_argument('--do_npz', default=False, action='store_true')


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

        if self.config['ema']:
            field_a = None
            field_b = self.generator
        else:
            field_a = self.generator
            field_b = None
        utils.load_weights(field_a, self.discriminator, state_dict,
                     self.config['weights_root'],
                     self.config['experiment_name'],
                     self.config['load_weights'],
                     field_b,
                     strict=False, load_optim=False)

        logging.log(logging.INFO, 'Weights loaded...')
        self.generator.to(self.config['device']).eval()
        self.discriminator.to(self.config['device']).eval()
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

    # parameterize config
    config['experiment_name'] = args.experiment_name
    config['ema'] = args.ema
    config['weights_root'] = args.weights_root
    config['smyrf'] = not args.disable_smyrf
    config['n_hashes'] = args.n_hashes
    config['q_cluster_size'] = args.q_cluster_size
    config['q_attn_size'] = args.q_cluster_size
    config['k_cluster_size'] = args.q_cluster_size // 4
    config['k_attn_size'] = args.q_cluster_size // 4
    config['dataset'] = args.dataset
    config['num_inception_images'] = args.num_inception_images

    biggan = Biggan(config)
    biggan.load_pretrained()

    # Random sampling
    category2indx = {val: key for key, val in indx2category.items()}
    if args.imagenet_category is not None:
        category = category2indx[args.imagenet_category]
    else:
        category = None

    if args.do_sample:
        for i in tqdm(range(args.sample_iters)):
            generator_inputs = biggan.get_random_inputs(bs=args.bs,
                                                        target=category,
                                                        seed=args.seed)
            out, _ = biggan.sample(generator_inputs, out_path='./samples.png')
            del generator_inputs, out, _

    if args.do_metrics:
        logging.log(logging.INFO, 'Preparing random sample sheet...')

        get_inception_metrics = inception_utils.prepare_inception_metrics(
            config['dataset'], parallel=True, no_fid=False)

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

    if args.do_npz:
        logging.log(logging.INFO, 'Doing npz for OFFICIAL scores...')
        x, y = [], []

        def sample():
            images, labels = biggan.sample(biggan.get_random_inputs(bs=args.bs))
            return images, labels

        for i in trange(int(np.ceil(config['num_inception_images'] / float(args.bs)))):
            with torch.no_grad():
                images, labels = sample()
                x.append(np.uint8(255 * (images.cpu().numpy() + 1) / 2.))
                y.append(labels.cpu().numpy())

        x = np.concatenate(x, 0)[:config['num_inception_images']]
        y = np.concatenate(y, 0)[:config['num_inception_images']]
        print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
        npz_filename = 'samples.npz'
        print('Saving npz to %s...' % npz_filename)
        np.savez(npz_filename, **{'x' : x, 'y' : y})
