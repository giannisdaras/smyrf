import torch
import torchvision.datasets as dsets
import torchvision
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import torch.utils.data as data
import torch_xla.core.xla_model as xm
import h5py as h5


class CelebAHQ(data.Dataset):
    def __init__(self, img_path, transform_img):
        self.img_path = img_path
        self.transform_img = transform_img
        self.data = []
        self.preprocess()
        self.num_images = len(self.data)

    def preprocess(self):
        length = len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])
        for i in range(length):
            img_path = os.path.join(self.img_path, str(i)+ '.jpg')
            self.data.append(img_path)
        xm.master_print('Finished preprocessing the CelebA dataset...')


    def __getitem__(self, index):
        dataset = self.data
        img_path = dataset[index]
        image = Image.open(img_path).convert('RGB')
        return self.transform_img(image), torch.tensor(0, dtype=torch.long)


    def __len__(self):
        """Return the number of images."""
        return self.num_images


'''
    ILSVRC_HDF5: A dataset to support I/O from an HDF5 to avoid
        having to load individual images all the time.
    Source code: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/datasets.py
'''
class ILSVRC_HDF5(data.Dataset):
  def __init__(self, root, transform=None, target_transform=None,
               load_in_mem=False, train=True,
               download=False, validate_seed=0,
               val_split=0, **kwargs): # last four are dummies

    self.root = root
    self.num_imgs = len(h5.File(root, 'r')['labels'])

    # self.transform = transform
    self.target_transform = target_transform

    # Set the transform here
    self.transform = transform

    # load the entire dataset into memory?
    self.load_in_mem = load_in_mem

    # If loading into memory, do so now
    if self.load_in_mem:
      print('Loading %s into memory...' % root)
      with h5.File(root,'r') as f:
        self.data = f['imgs'][:]
        self.labels = f['labels'][:]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    # If loaded the entire dataset in RAM, get image from memory
    if self.load_in_mem:
      img = self.data[index]
      target = self.labels[index]

    # Else load it from disk
    else:
      with h5.File(self.root,'r') as f:
        img = f['imgs'][index]
        target = f['labels'][index]

    img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, int(target)

  def __len__(self):
      return self.num_imgs
