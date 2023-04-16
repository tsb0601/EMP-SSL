import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import torchvision
# import torch.nn


from torch import nn, optim
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader



from typing import Tuple


import torch.nn.functional as F
from torchmetrics.metric import Metric


class WeightedKNNClassifier(Metric):
    def __init__(
        self,
        k: int = 20,
        T: float = 0.07,
        max_distance_matrix_size: int = int(5e6),
        distance_fx: str = "cosine",
        epsilon: float = 0.00001,
        dist_sync_on_step: bool = False,
    ):
        """Implements the weighted k-NN classifier used for evaluation.
        Args:
            k (int, optional): number of neighbors. Defaults to 20.
            T (float, optional): temperature for the exponential. Only used with cosine
                distance. Defaults to 0.07.
            max_distance_matrix_size (int, optional): maximum number of elements in the
                distance matrix. Defaults to 5e6.
            distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
                "euclidean". Defaults to "cosine".
            epsilon (float, optional): Small value for numerical stability. Only used with
                euclidean distance. Defaults to 0.00001.
            dist_sync_on_step (bool, optional): whether to sync distributed values at every
                step. Defaults to False.
        """

        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.k = k
        self.T = T
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon

        self.add_state("train_features", default=[], persistent=False)
        self.add_state("train_targets", default=[], persistent=False)
        self.add_state("test_features", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)

    def update(
        self,
        train_features: torch.Tensor = None,
        train_targets: torch.Tensor = None,
        test_features: torch.Tensor = None,
        test_targets: torch.Tensor = None,
    ):
        """Updates the memory banks. If train (test) features are passed as input, the
        corresponding train (test) targets must be passed as well.
        Args:
            train_features (torch.Tensor, optional): a batch of train features. Defaults to None.
            train_targets (torch.Tensor, optional): a batch of train targets. Defaults to None.
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    def set_tk(self, T, k):
        self.T = T
        self.k = k
        
    @torch.no_grad()
    def compute(self) -> Tuple[float]:
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.
        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """
        
        #print(self.T, self.k)

        train_features = torch.cat(self.train_features)
        train_targets = torch.cat(self.train_targets)
        test_features = torch.cat(self.test_features)
        test_targets = torch.cat(self.test_targets)

        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)
        num_train_images = train_targets.size(0)
        chunk_size = min(
            max(1, self.max_distance_matrix_size // num_train_images),
            num_test_images,
        )
        k = min(self.k, num_train_images)

        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in range(0, num_test_images, chunk_size):
            # get the features for test images
            features = test_features[idx : min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                similarities = torch.mm(features, train_features.t())
            elif self.distance_fx == "euclidean":
                similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
            else:
                raise NotImplementedError

            similarities, indices = similarities.topk(k, largest=True, sorted=True)
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                similarities = similarities.clone().div_(self.T).exp_()

            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    similarities.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = (
                top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
            )  # top5 does not make sense if k < 5
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total

        self.reset()

        return top1, top5







def linear(train_features, train_labels, test_features, test_labels, lr=0.0075, num_classes = 100):


   
    
    train_data = tensor_dataset(train_features,train_labels)
    test_data = tensor_dataset(test_features,test_labels)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=True, drop_last=False, num_workers=2)
    
    LL = nn.Linear(train_features.shape[1],num_classes)
    optimizer = torch.optim.SGD(LL.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    test_acc_list = []
    for epoch in range(100):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch
            y_batch = y_batch
            
            logits = LL(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step() 

        top1_train_accuracy /= (counter + 1)

        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch
            y_batch = y_batch

            logits = LL(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        
        test_acc_list.append(top1_accuracy)
        
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    acc_vect = torch.tensor(test_acc_list)
    print('best linear test acc {}, last acc {}'.format(acc_vect.max().item(),acc_vect[-1].item()))
        


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
            
class tensor_dataset(data.Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.length = x.shape[0]
    
    def __getitem__(self,indx):
        return self.x[indx], self.y[indx]
    
    def __len__(self):
        return self.length




def set_gamma(loss_fn,epoch,total_epoch=500,warmup_epoch=100,gamma_min=0.,gamma_max=1.0):
    warmup_start = total_epoch - warmup_epoch
    warmup_end = total_epoch
    
    if warmup_start < epoch<=warmup_end:
        loss_fn.gamma = ((epoch - warmup_start)/(warmup_end - warmup_start))*(gamma_max - gamma_min) + gamma_min
    else:
        loss_fn.gamma = gamma_min

def warmup_lr(optimizer,epoch,base_lr,warmup_epoch=10):
    if epoch<warmup_epoch:
        optimizer.param_groups[0]['lr'] = base_lr*min(1.,(epoch+1)/warmup_epoch)
        
        
def marginal_H(logits):
    bs = torch.tensor(logits.shape[0]).float()
    logps = torch.log_softmax(logits,dim=1)
    marginal_p = torch.logsumexp(logps - bs.log(),dim=0)
    H = (marginal_p.exp()*(-marginal_p)).sum()*(1.4426950)
    return H

def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)

def cluster_match(cluster_mtx,label_mtx,n_classes=10,print_result=True):
    #verified to be consistent to optimimal assignment problem based algorithm
    cluster_indx = list(cluster_mtx.unique())
    assigned_label_list = []
    assigned_count = []
    while (len(assigned_label_list)<=n_classes) and len(cluster_indx)>0:
        max_label_list = []
        max_count_list = []
        for indx in cluster_indx:
            #calculate highest number of matchs
            mask = cluster_mtx==indx
            label_elements, counts = label_mtx[mask].unique(return_counts=True)
            for assigned_label in assigned_label_list:
                counts[label_elements==assigned_label] = 0
            max_count_list.append(counts.max())
            max_label_list.append(label_elements[counts.argmax()])

        max_label = torch.stack(max_label_list)
        max_count = torch.stack(max_count_list)
        assigned_label_list.append(max_label[max_count.argmax()])
        assigned_count.append(max_count.max())
        cluster_indx.pop(max_count.argmax())
    total_correct = torch.tensor(assigned_count).sum().item()
    total_sample = cluster_mtx.shape[0]
    acc = total_correct/total_sample
    if print_result:
        print('{}/{} ({}%) correct'.format(total_correct,total_sample,acc*100))
    else:
        return total_correct, total_sample, acc

def cluster_merge_match(cluster_mtx,label_mtx,print_result=True):
    cluster_indx = list(cluster_mtx.unique())
    n_correct = 0
    for cluster_id in cluster_indx:
        label_elements, counts = label_mtx[cluster_mtx==cluster_id].unique(return_counts=True)
        n_correct += counts.max()
    total_sample = len(cluster_mtx)
    acc = n_correct.item()/total_sample
    if print_result:
        print('{}/{} ({}%) correct'.format(n_correct,total_sample,acc*100))
    else:
        return n_correct, total_sample, acc

    
def cluster_acc(test_loader,net,device,print_result=False,save_name_img='cluster_img',save_name_fig='pca_figure'):
    cluster_list = []
    label_list = []
    x_list = []
    z_list = []
    net.eval()
    for x, y in test_loader:
        with torch.no_grad():
            x, y = x.float().to(device), y.to(device)
            z, logit = net(x)
            if logit.sum() == 0:
                logit += torch.randn_like(logit)
            cluster_list.append(logit.max(dim=1)[1].cpu())
            label_list.append(y.cpu())
            x_list.append(x.cpu())
            z_list.append(z.cpu())
    net.train()
    cluster_mtx = torch.cat(cluster_list,dim=0)
    label_mtx = torch.cat(label_list,dim=0)
    x_mtx = torch.cat(x_list,dim=0)
    z_mtx = torch.cat(z_list,dim=0)
    _, _, acc_single = cluster_match(cluster_mtx,label_mtx,n_classes=label_mtx.max()+1,print_result=False)
    _, _, acc_merge = cluster_merge_match(cluster_mtx,label_mtx,print_result=False)
    NMI = normalized_mutual_info_score(label_mtx.numpy(),cluster_mtx.numpy())
    ARI = adjusted_rand_score(label_mtx.numpy(),cluster_mtx.numpy())
    if print_result:
        print('cluster match acc {}, cluster merge match acc {}, NMI {}, ARI {}'.format(acc_single,acc_merge,NMI,ARI))
    
    save_name_img += '_acc'+ str(acc_single)[2:5]
    save_cluster_imgs(cluster_mtx,x_mtx,save_name_img)
    save_latent_pca_figure(z_mtx,cluster_mtx,save_name_fig)
    
    return acc_single, acc_merge, NMI, ARI
    
def save_cluster_imgs(cluster_mtx,x_mtx,save_name,npercluster=100):
    cluster_indexs, counts = cluster_mtx.unique(return_counts=True)
    x_list = []
    counts_list = []
    for i, c_indx in enumerate(cluster_indexs):
        if counts[i]>npercluster:
            x_list.append(x_mtx[cluster_mtx==c_indx,:,:,:])
            counts_list.append(counts[i])

    n_clusters = len(counts_list)
    fig, ax = plt.subplots(n_clusters,1,dpi=80,figsize=(1.2*n_clusters, 3*n_clusters))
    for i, ax in enumerate(ax):
        img = torchvision.utils.make_grid(x_list[i][:npercluster],nrow=npercluster//5,normalize=True)
        ax.imshow(img.permute(1,2,0))
        ax.set_axis_off()

        ax.set_title('Cluster with {} images'.format(counts_list[i]))
    
    fig.savefig(save_name+'.pdf')
    plt.close(fig)
    
def save_latent_pca_figure(z_mtx,cluster_mtx,save_name):
    _, s_z_all, _ = z_mtx.svd()
    cluster_n = []
    cluster_s = []
    for cluster_indx in cluster_mtx.unique():
        _, s_cluster, _ = z_mtx[cluster_mtx==cluster_indx,:].svd()
        cluster_n.append((cluster_mtx==cluster_indx).sum().item())
        cluster_s.append(s_cluster/s_cluster.max())

    #make plot
    fig, ax = plt.subplots(1,2,figsize=(9, 3))
    ax[0].plot(s_z_all)
    for i, s_curve in enumerate(cluster_s):
        ax[1].plot(s_curve,label=cluster_n[i])
    ax[1].set_xlim(xmin=0,xmax=20)
    ax[1].legend()
    fig.savefig(save_name +'.pdf')
    plt.close(fig)
    
def analyze_latent(z_mtx,cluster_mtx):
    _, s_z_all, _ = z_mtx.svd()
    cluster_n = []
    cluster_s = []
    cluster_d = []
    for cluster_indx in cluster_mtx.unique():
        _, s_cluster, _ = z_mtx[cluster_mtx==cluster_indx,:].svd()
        s_cluster = s_cluster/s_cluster.max()
        cluster_n.append((cluster_mtx==cluster_indx).sum().item())
        cluster_s.append(s_cluster)
#         print(list(cluster_s))
        print(s_cluster)
#         s_diff = s_cluster[:-1] - s_cluster[1:]
#         cluster_d.append(s_diff.max(0)[1])
        cluster_d.append((s_cluster>0.01).sum())
    for i in range(len(cluster_n)):
        print('subspace {}, dimension {}, samples {}'.format(i,cluster_d[i],cluster_n[i]))