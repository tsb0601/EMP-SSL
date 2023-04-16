############
## Import ##
############
import argparse
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from model.model import encoder
from dataset.datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as FF
from tqdm import tqdm
import torch
from torchvision.datasets import CIFAR10
from loss import TotalCodingRate
from func import chunk_avg
from lars import LARS, LARSWrapper
from func import WeightedKNNClassifier
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast

######################
## Parsing Argument ##
######################
import argparse
parser = argparse.ArgumentParser(description='Unsupervised Learning')

parser.add_argument('--patch_sim', type=int, default=200,
                    help='coefficient of cosine similarity (default: 200)')
parser.add_argument('--tcr', type=int, default=1,
                    help='coefficient of tcr (default: 1)')
parser.add_argument('--num_patches', type=int, default=100,
                    help='number of patches used in EMP-SSL (default: 100)')
parser.add_argument('--arch', type=str, default="resnet18-cifar",
                    help='network architecture (default: resnet18-cifar)')
parser.add_argument('--bs', type=int, default=100,
                    help='batch size (default: 100)')
parser.add_argument('--lr', type=float, default=0.3,
                    help='learning rate (default: 0.3)')        
parser.add_argument('--eps', type=float, default=0.2,
                    help='eps for TCR (default: 0.2)') 
parser.add_argument('--msg', type=str, default="NONE",
                    help='additional message for description (default: NONE)')     
parser.add_argument('--dir', type=str, default="EMP-SSL-Training",
                    help='directory name (default: EMP-SSL-Training)')     
parser.add_argument('--data', type=str, default="cifar10",
                    help='data (default: cifar10)')          
parser.add_argument('--epoch', type=int, default=30,
                    help='max number of epochs to finish (default: 30)')  

args = parser.parse_args()

print(args)

num_patches = args.num_patches
dir_name = f"./logs/{args.dir}/patchsim{args.patch_sim}_numpatch{args.num_patches}_bs{args.bs}_lr{args.lr}_{args.msg}"



#####################
## Helper Function ##
#####################

def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)


class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
            
        z_sim = z_sim/num_patch
        z_sim_out = z_sim.clone().detach()
                
        return -z_sim, z_sim_out
    
def cal_TCR(z, criterion, num_patches):
    z_list = z.chunk(num_patches,dim=0)
    loss = 0
    for i in range(num_patches):
        loss += criterion(z_list[i])
    loss = loss/num_patches
    return loss

######################
## Prepare Training ##
######################
torch.multiprocessing.set_sharing_strategy('file_system')

if args.data == "imagenet100" or args.data == "imagenet":
    train_dataset = load_dataset("imagenet", train=True, num_patch = num_patches)
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=8)

else:
    train_dataset = load_dataset(args.data, train=True, num_patch = num_patches)
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=16)


use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
    
    
net = encoder(arch = args.arch)
net = nn.DataParallel(net)
net.cuda()


opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,nesterov=True)
opt = LARSWrapper(opt,eta=0.005,clip=True,exclude_bias_n_norm=True,)

scaler = GradScaler()
if args.data == "imagenet-100":
    num_converge = (150000//args.bs)*args.epoch
else:
    num_converge = (50000//args.bs)*args.epoch
    
scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=num_converge, eta_min=0,last_epoch=-1)

# Loss
contractive_loss = Similarity_Loss()
criterion = TotalCodingRate(eps=args.eps)


##############
## Training ##
##############
def main():
    for epoch in range(args.epoch):            
        for step, (data, label) in tqdm(enumerate(dataloader)):
            net.zero_grad()
            opt.zero_grad()
        
            data = torch.cat(data, dim=0) 
            data = data.cuda()
            z_proj = net(data)
            
            z_list = z_proj.chunk(num_patches, dim=0)
            z_avg = chunk_avg(z_proj, num_patches)
            
            
            #Contractive Loss
            loss_contract, _ = contractive_loss(z_list, z_avg)
            loss_TCR = cal_TCR(z_proj, criterion, num_patches)
            
            loss = args.patch_sim*loss_contract + args.tcr*loss_TCR
          
            loss.backward()
            opt.step()
            scheduler.step()
            

        model_dir = dir_name+"/save_models/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(net.state_dict(), model_dir+str(epoch)+".pt")
        
    
        print("At epoch:", epoch, "loss similarity is", loss_contract.item(), ",loss TCR is:", (loss_TCR).item(), "and learning rate is:", opt.param_groups[0]['lr'])
       
                


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
