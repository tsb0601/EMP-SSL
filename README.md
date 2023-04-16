We will release the code very soon~~~ Sorry the server was down (and still down sadly :<) We will release the code asap when the server back on



# EMP-SSL: Towards Self-Supervised Learning in One Training Epoch

[![arXiv](https://img.shields.io/badge/arXiv-2304.03977-b31b1b.svg)](https://arxiv.org/abs/2304.03977)


![Training Pipeline](pipeline.png)


Authors: Shengbang Tong, Yubei Chen, Yi Ma, Yann Lecun

## Introduction
This repository contains the implementation for the paper "EMP-SSL: Towards Self-Supervised Learning in One Training Epoch." The paper introduces a simplistic but efficient self-supervised learning method called Extreme-Multi-Patch Self-Supervised-Learning (EMP-SSL). EMP-SSL significantly reduces the training epochs required for convergence by increasing the number of fix size image patches from each image instance.

## Getting Started

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

## Acknowledgment: 
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