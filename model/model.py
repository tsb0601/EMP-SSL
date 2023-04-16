import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50

from .resnet import Resnet10CIFAR

def getmodel(arch):
    
    #backbone = resnet18()
    
    if arch == "resnet18-cifar":
        backbone = resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        return backbone, 512  
    elif arch == "resnet18-imagenet":
        backbone = resnet18()    
        backbone.fc = nn.Identity()
        return backbone, 512
    elif arch == "resnet18-tinyimagenet":
        backbone = resnet18()    
        backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        backbone.fc = nn.Identity()
        return backbone, 512
    else:
        raise NameError("{} not found in network architecture".format(arch))
  

class encoder(nn.Module): 
     def __init__(self,z_dim=1024,hidden_dim=4096, norm_p=2, arch = "resnet18-cifar"):
        super().__init__()

        backbone, feature_dim = getmodel(arch)
        self.backbone = backbone
        self.norm_p = norm_p
        self.pre_feature = nn.Sequential(nn.Linear(feature_dim,hidden_dim),
                                         nn.BatchNorm1d(hidden_dim),
                                         nn.ReLU()
                                        )
        self.projection = nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,z_dim))
        
          
     def forward(self, x, is_test = False):
         
        feature = self.backbone(x)
        feature = self.pre_feature(feature)
        z = F.normalize(self.projection(feature),p=self.norm_p)

        if is_test:
            return z, feature
        else:
            return z

   
    