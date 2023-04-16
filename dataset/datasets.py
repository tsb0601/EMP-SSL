import os
import numpy as np
import torchvision

def load_dataset(data_name, train=True, num_patch = 4, path="./data/"):
    """Loads a dataset for training and testing. If augmentloader is used, transform should be None.
    
    Parameters:
        data_name (str): name of the dataset
        transform_name (torchvision.transform): name of transform to be applied (see aug.py)
        use_baseline (bool): use baseline transform or augmentation transform
        train (bool): load training set or not
        contrastive (bool): whether to convert transform to multiview augmentation for contrastive learning.
        n_views (bool): number of views for contrastive learning
        path (str): path to dataset base path

    Returns:
        dataset (torch.data.dataset)
    """
    _name = data_name.lower()
    if _name == "imagenet":
        from .aug4img import ContrastiveLearningViewGenerator
    else:
        from .aug import ContrastiveLearningViewGenerator
      
    
    transform = ContrastiveLearningViewGenerator(num_patch = num_patch)
        
    if _name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(path, "CIFAR10"), train=train, download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "CIFAR100"), train=train, download=True, transform=transform)
        trainset.num_classes = 100
    elif _name == "imagenet":
        if train:
            trainset = torchvision.datasets.ImageFolder(root="/home/peter/Data/ILSVRC2012/train100/",transform=transform)
            #trainset = torchvision.datasets.ImageFolder(root="/home/peter/Data/tiny-imagenet-200/train/",transform=transform)
        else:
            trainset = torchvision.datasets.ImageFolder(root="/home/peter/Data/ILSVRC2012/val100/",transform=transform)
            #trainset = torchvision.datasets.ImageFolder(root="/home/peter/Data/tiny-imagenet-200/val/",transform=transform)
        trainset.num_classes = 200  
        
    else:
        raise NameError("{} not found in trainset loader".format(_name))
    return trainset

def sparse2coarse(targets):
    """CIFAR100 Coarse Labels. """
    coarse_targets = [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,
                       9,  7, 11,  6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10,
                      12, 14, 16,  9, 11,  5,  5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,
                       4, 17,  4,  2,  0, 17,  4, 18, 17, 10,  3,  2, 12, 12, 16, 12,  1,
                       9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 16, 19,  2,  4,  6,
                      19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
    return np.array(coarse_targets)[targets]