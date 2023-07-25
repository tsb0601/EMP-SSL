# EMP-SSL: Towards Self-Supervised Learning in One Training Epoch

[![arXiv](https://img.shields.io/badge/arXiv-2304.03977-b31b1b.svg)](https://arxiv.org/abs/2304.03977)


![Training Pipeline](pipeline.png)


Authors: Shengbang Tong*, Yubei Chen*, Yi Ma, Yann LeCun

## Introduction
This repository contains the implementation for the paper "EMP-SSL: Towards Self-Supervised Learning in One Training Epoch." The paper introduces a simplistic but efficient self-supervised learning method called Extreme-Multi-Patch Self-Supervised-Learning (EMP-SSL). EMP-SSL significantly reduces the training epochs required for convergence by increasing the number of fix size image patches from each image instance.

## Preparing Training Data
Cifar10 and Cifar100 can be downloaded automatically in the script. ImageNet100 is a special subset of ImageNet. Details can be found in this [link](https://github.com/HobbitLong/CMC/issues/21).

## Getting Started
Current code implementation supports Cifar10, Cifar100 and ImageNet100.

To get started with the EMP-SSL implementation, follow these instructions:

### 1. Clone this repository
```bash
git clone https://github.com/tsb0601/emp-ssl.git
cd emp-ssl
``` 
### 2. Install required packages
```
pip install -r requirements.txt
```
### 3. Training

#### Reproducing 1-epoch results

|                    | CIFAR-10<br>1 Epoch | CIFAR-100<br>1 Epoch | Tiny ImageNet<br>1 epochs | ImageNet-100<br>1 epochs |
|--------------------|:----------------------:|:-----------------------:|:----------------------------:|:--------------------------:|
| EMP-SSL (1 Epoch)  |         0.842          |          0.585          |             0.381             |            0.585           |

For CIFAR10 or CIFAR100
```
python main.py --data cifar10 --epoch 2 --patch_sim 200 --arch 'resnet18-cifar' --num_patches 20 --lr 0.3
```
For ImageNet100
```
python main.py --data imagenet100 --epoch 2 --patch_sim 200 --arch 'resnet18-imagenet' --num_patches 20 --lr 0.3
```


#### Reproducing multi epochs results

|                      | CIFAR-10<br>1 Epoch | CIFAR-10<br>10 Epochs | CIFAR-10<br>30 Epochs | CIFAR-10<br>1000 Epochs | CIFAR-100<br>1 Epoch | CIFAR-100<br>10 Epochs | CIFAR-100<br>30 Epochs | CIFAR-100<br>1000 Epochs | Tiny ImageNet<br>10 Epochs | Tiny ImageNet<br>1000 Epochs |ImageNet-100<br>10 Epochs | ImageNet-100<br>400 Epochs |
|----------------------|:-------------------:|:---------------------:|:---------------------:|:-----------------------:|:--------------------:|:----------------------:|:----------------------:|:------------------------:| :------------------------:|:------------------------:|:------------------------:| :------------------------:|
| SimCLR               |        0.282        |         0.565         |         0.663         |          0.910          |         0.054        |         0.185          |         0.341          |          0.662           | - | 0.488 | - | 0.776
| BYOL                 |        0.249        |         0.489         |         0.684         |          0.926          |         0.043        |         0.150          |         0.349          |          0.708           | - | 0.510 | - | 0.802
| VICReg               |        0.406        |         0.697         |         0.781         |          0.921          |         0.079        |         0.319          |         0.479          |          0.685           | - | - | - | 0.792
| SwAV                 |        0.245        |         0.532         |         0.767         |          0.923          |         0.028        |         0.208          |         0.294          |          0.658           |- | - | - | 0.740
| ReSSL                |        0.245        |         0.256         |         0.525         |          0.914          |         0.033        |         0.122          |         0.247          |          0.674           |- | - | - | 0.769
| EMP-SSL (20 patches) |        0.806        |         0.907         |         0.931         |            -            |         0.551        |         0.678          |         0.724          |            -              | - | - | - | -
| EMP-SSL (200 patches)|        0.826        |         0.915         |         0.934         |            -            |         0.577        |         0.701          |         0.733          |            -              | 0.515 | - | 0.789 | -

Change num_patches here to change the number of patches used in EMP-SSL training.
```
python main.py --data cifar10 --epoch 30 --patch_sim 200 --arch 'resnet18-cifar' --num_patches 20 --lr 0.3
```



### 4. Evaluating
Because our model is trained with only fixed size image patches. To evaluate the performance, we adopt bag-of-features model from intra-instance VICReg paper. Change test_patches here to adjust number of patches used in bag-of-feature model for different GPUs.
```
python evaluate.py --model_path 'path to your evaluated model' --test_patches 128
```

## Acknowledgment
This repo is inspired by [MCR2](https://github.com/Ma-Lab-Berkeley/MCR2), [solo-learn](https://github.com/vturrisi/solo-learn) and [NMCE](https://github.com/zengyi-li/NMCE-release) repo.

## Citation
If you find this repository useful, please consider giving a star :star: and citation:

```
@article{tong2023empssl,
title={EMP-SSL: Towards Self-Supervised Learning in One Training Epoch},
author={Shengbang Tong and Yubei Chen and Yi Ma and Yann Lecun},
journal={arXiv preprint arXiv:2304.03977},
year={2023}
}
```
