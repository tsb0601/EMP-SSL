import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34, resnet50

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_config, first_config, first_pool=False):
        super(ResNet, self).__init__()
        #format of first_config
        [in_chan, chan, k, s] = first_config
        self.in_planes = chan
        self.conv1 = nn.Conv2d(in_chan, chan, kernel_size=k, stride=s,
                               padding=k//2, bias=False)
        self.bn1 = nn.BatchNorm2d(chan)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if first_pool else nn.Identity()
        self.layer1 = self._make_layer(block, blocks_config[0][0], blocks_config[0][1], stride=1)
        self.layer2 = self._make_layer(block, blocks_config[1][0], blocks_config[1][1], stride=2)
        self.layer3 = self._make_layer(block, blocks_config[2][0], blocks_config[2][1], stride=2)
        self.layer4 = self._make_layer(block, blocks_config[3][0], blocks_config[3][1], stride=2)
    
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        
        
        feature = out.mean((2,3))
        
        return feature
    

def Resnet10MNIST():
    block = BasicBlock
    blocks_config = [
        [64,1],[128,1],[256,1],[512,1]
    ]
    first_config = [1,64,3,1]
    return ResNet(block,blocks_config,first_config,first_pool=False)

def Resnet10CIFAR():
    block = BasicBlock
    blocks_config = [
        [32,1],[64,1],[128,1],[256,1] 
    ]
    first_config = [3,32,3,1]
    return ResNet(block,blocks_config,first_config,first_pool=True)

def Resnet18imgs():
    block = BasicBlock
    blocks_config = [
        [32,2],[64,2],[128,2],[256,2]
    ]
    first_config = [1,32,5,2]
    return ResNet(block,blocks_config,first_config,first_pool=True)

def Resnet18CIFAR():
    backbone = resnet18()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    return backbone
    
def Resnet18STL10():
    block = BasicBlock
    blocks_config = [
        [64,2],[128,2],[256,2],[512,2]
    ]
    first_config = [3,64,5,2]
    return ResNet(block,blocks_config,first_config,first_pool=True)

def Resnet34CIFAR():
    backbone = resnet34()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    return backbone

def Resnet34STL10():
    block = BasicBlock
    blocks_config = [
        [64,3],[128,4],[256,6],[512,3]
    ]
    first_config = [3,64,5,2]
    return ResNet(block,blocks_config,first_config,first_pool=True)