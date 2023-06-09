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
For CIFAR10 or CIFAR100
```
python main.py --data cifar10 --epoch 2 --patch_sim 200 --arch 'resnet18-cifar' --num_patches 20 --lr 0.3
```
For ImageNet100
```
python main.py --data imagenet100 --epoch 2 --patch_sim 200 --arch 'resnet18-imagenet' --num_patches 20 --lr 0.3
```


#### Reproducing multi epochs results
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
