import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps


def load_transforms(name):
    """Load data transformations.
    
    Note:
        - Gaussian Blur is defined at the bottom of this file.
    
    """
    _name = name.lower()
    if _name == "cifar_sup":
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32,scale=(0.765625, 0.765625),ratio=(1., 1.)),
            transforms.RandomCrop(32, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        baseline_transform = transforms.Compose([
            transforms.RandomResizedCrop(32,scale=(0.765625, 0.765625),ratio=(1., 1.)),
            transforms.ToTensor(),normalize])

    elif _name == "cifar_patch":
        normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        aug_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.ToTensor(),
            normalize
        ])
        baseline_transform = transforms.Compose([
            transforms.ToTensor(), normalize])
        
    elif _name == "cifar_simclr_norm":
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32,scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        baseline_transform = transforms.Compose([
            transforms.ToTensor(),normalize])
    
    elif _name == "cifar_byol":
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                    (32, 32),
                    scale=(0.2, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        baseline_transform = transforms.Compose([
#             transforms.RandomResizedCrop(32,scale=(0.765625, 0.765625),ratio=(1., 1.)),
            transforms.ToTensor(),normalize])

    else:
        raise NameError("{} not found in transform loader".format(name))
    return aug_transform, baseline_transform


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)

class GBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ContrastiveLearningViewGenerator(object):
    def __init__(self, num_patch = 4):
    
        self.num_patch = num_patch
      
    def __call__(self, x):
    
    
        normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32,scale=(0.25, 0.25), ratio=(1,1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),  
            normalize
        ])
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]
     
        return augmented_x
