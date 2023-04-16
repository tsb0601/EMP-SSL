import torch
import torch.nn as nn
import torch.nn.functional as F

class contrastive_loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,x,labels):
        #this function assums that positive logit is always the first element.
        #Which is true here
        loss = -x[:,0] + torch.logsumexp(x[:,1:],dim=1)
        return loss.mean()

class SimCLR(nn.Module):
    def __init__(self,temperature=0.5,n_views=2,contrastive=False):
        super(SimCLR,self).__init__()
        self.temp = temperature
        self.n_views = n_views
        
        if contrastive:
            self.criterion = contrastive_loss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        
    def info_nce_loss(self,X):
        
        bs, n_dim = X.shape
        bs = int(bs/self.n_views)
        device = X.device
        
        
        labels = torch.cat([torch.arange(bs) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        similarity_matrix = torch.matmul(X, X.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        
        logits = logits / self.temp
        return logits, labels
        
    def forward(self,X):
        logits, labels = self.info_nce_loss(X)
        loss = self.criterion(logits, labels)
        return loss

class Z_loss(nn.Module):
    def __init__(self,):
        super().__init__()
        pass
        
    def forward(self,z):
        z_list = z.chunk(2,dim=0)
        z_sim = F.cosine_similarity(z_list[0],z_list[1],dim=1).mean()
        z_sim_out = z_sim.clone().detach()
        return -z_sim, z_sim_out

class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)

class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, eps=0.01, gamma=1):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma = gamma
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def compute_compress_loss(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p,device=W.device).expand((k,p,p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p/(trPi*self.eps)).view(k,1,1)
        
        W = W.view((1,p,m))
        log_det = torch.logdet(I + scale*W.mul(Pi).matmul(W.transpose(1,2)))
        compress_loss = (trPi.squeeze()*log_det/(2*m)).sum()
        return compress_loss
        
    def forward(self, X, Y, num_classes=None):
        #This function support Y as label integer or membership probablity.
        if len(Y.shape)==1:
            #if Y is a label vector
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes,1,Y.shape[0]),device=Y.device)
            for indx, label in enumerate(Y):
                Pi[label,0,indx] = 1
        else:
            #if Y is a probility matrix
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes,1,-1))
            
        W = X.T
        discrimn_loss = self.compute_discrimn_loss(W)
        compress_loss = self.compute_compress_loss(W, Pi)
 
        total_loss = - discrimn_loss + self.gamma*compress_loss
        return total_loss, [discrimn_loss.item(), compress_loss.item()]