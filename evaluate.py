############
## Import ##
############
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import encoder
from dataset.datasets import load_dataset
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch
import numpy as np
from func import WeightedKNNClassifier, linear

######################
## Parsing Argument ##
######################
import argparse
parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--test_patches', type=int, default=128,
                    help='number of patches used in testing (default: 128)')  

parser.add_argument('--data', type=str, default="cifar10",
                    help='dataset (default: cifar10)')  
parser.add_argument('--arch', type=str, default="resnet18-cifar",
                    help='network architecture (default: resnet18-cifar)')

parser.add_argument('--lr', type=float, default=0.03,
                    help='learning rate for linear eval (default: 0.03)')        
parser.add_argument('--linear', type=bool, default=True,
                    help='use linear eval or not')
parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')
parser.add_argument('--model_path', type=str, default="",
                    help='model directory for eval')

            
args = parser.parse_args()















######################
## Testing Accuracy ##
######################
test_patches = args.test_patches

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

knn_classifier = WeightedKNNClassifier()


def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)


def test(net, train_loader, test_loader):
    
    train_z_full_list, train_y_list, test_z_full_list, test_y_list = [], [], [], []
    
    with torch.no_grad():
        for x, y in tqdm(train_loader):

            x = torch.cat(x, dim = 0)
            
            z_proj, z_pre = net(x, is_test=True)

            z_pre = chunk_avg(z_pre, test_patches)
            z_pre = z_pre.detach().cpu()
            
            
            train_z_full_list.append(z_pre)
            
            
            knn_classifier.update(train_features = z_pre, train_targets = y)

            train_y_list.append(y)
                
        for x, y in tqdm(test_loader):
            x = torch.cat(x, dim = 0)
            
            z_proj, z_pre = net(x, is_test=True)

            z_pre = chunk_avg(z_pre, test_patches)
            z_pre = z_pre.detach().cpu()
           
            test_z_full_list.append(z_pre)
       
            knn_classifier.update(test_features = z_pre, test_targets = y)

            test_y_list.append(y)
                
            
    train_features_full, train_labels, test_features_full, test_labels = torch.cat(train_z_full_list,dim=0), torch.cat(train_y_list,dim=0), torch.cat(test_z_full_list,dim=0), torch.cat(test_y_list,dim=0)
   
    if args.data == "cifar10":
        num_classes = 10
    elif args.data == "cifar100":
        num_classes = 100
    elif args.data == "tinyimagenet200":
        num_classes = 200
    elif args.data == "imagenet100":
        num_classes = 100
    elif args.data == "imagenet":
        num_classes = 1000
        
    if args.linear:
        print("Using Linear Eval to evaluate accuracy")
        linear(train_features_full, train_labels, test_features_full, test_labels, lr=args.lr, num_classes = num_classes)
    
    if args.knn:
        print("Using KNN to evaluate accuracy")
        top1, top5 = knn_classifier.compute()
        print("KNN (top1/top5):", top1, top5)
    
def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)


torch.multiprocessing.set_sharing_strategy('file_system')


#Get Dataset
if args.data == "imagenet100" or args.data == "imagenet":
        
    memory_dataset = load_dataset(args.data, train=True, num_patch = test_patches)
    memory_loader = DataLoader(memory_dataset, batch_size=50, shuffle=True, drop_last=True,num_workers=8)

    test_data = load_dataset(args.data, train=False, num_patch = test_patches)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=8)

else:
    memory_dataset = load_dataset(args.data, train=True, num_patch = test_patches)
    memory_loader = DataLoader(memory_dataset, batch_size=50, shuffle=True, drop_last=True,num_workers=8)

    test_data = load_dataset(args.data, train=False, num_patch = test_patches)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=8)

# Load Model and Checkpoint
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
net = encoder(arch = args.arch)
net = nn.DataParallel(net)
save_dict = torch.load(args.model_path)
net.load_state_dict(save_dict,strict=False)
net.cuda()
net.eval()
test(net, memory_loader, test_loader)



