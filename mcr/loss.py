import torch
import torch.nn as nn
import torch.nn.functional as F


class MCRGANloss(nn.Module):

    def __init__(self, gam1=1., gam2=1., gam3=1., eps=0.5, numclasses=1000, mode=0):
        super(MCRGANloss, self).__init__()

        self.num_class = numclasses
        self.train_mode = mode
        self.gam1 = gam1
        self.gam2 = gam2
        self.gam3 = gam3
        self.eps = eps

    def forward(self, Z, Z_bar, real_label, ith_inner_loop, num_inner_loop):

        # t = time.time()
        errD, empi = self.old_version(Z, Z_bar, real_label, ith_inner_loop, num_inner_loop)

        return errD, empi

    def old_version(self, Z, Z_bar, real_label, ith_inner_loop, num_inner_loop):

        if self.train_mode == 2:
            loss_z, _ = self.deltaR(Z, real_label, self.num_class)
            assert num_inner_loop >= 2
            if (ith_inner_loop + 1) % num_inner_loop != 0:
                # print(f"{ith_inner_loop + 1}/{num_inner_loop}")
                # print("calculate delta R(z)")
                return loss_z, None

            loss_h, _ = self.deltaR(Z_bar, real_label, self.num_class)

            empi = [loss_z, loss_h]
            term3 = 0.
            for i in range(self.num_class):
                new_Z = torch.cat((Z[real_label == i], Z_bar[real_label == i]), 0)
                new_label = torch.cat(
                    (torch.zeros_like(real_label[real_label == i]),
                     torch.ones_like(real_label[real_label == i]))
                )
                loss, em = self.deltaR(new_Z, new_label, 2)
                term3 += loss
            empi = empi + [term3]
            errD = self.gam1 * loss_z + self.gam2 * loss_h + self.gam3 * term3

        elif self.train_mode == 1:
            print("has been dropped")
            raise NotImplementedError()

        elif self.train_mode == 0:
            new_Z = torch.cat((Z, Z_bar), 0)
            new_label = torch.cat((torch.zeros_like(real_label), torch.ones_like(real_label)))
            errD, empi = self.deltaR(new_Z, new_label, 2)
        else:
            raise ValueError()

        return errD, empi

    def debug(self, Z, Z_bar, real_label):

        print("===========================")

    def compute_discrimn_loss(self, Z):
        """Theoretical Discriminative Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        scalar = d / (n * self.eps)
        logdet = torch.logdet(I + scalar * Z @ Z.T)
        return logdet / 2.

    def compute_compress_loss(self, Z, Pi):
        """Theoretical Compressive Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = []
        scalars = []
        for j in range(Pi.shape[1]):
            Z_ = Z[:, Pi[:, j] == 1]
            trPi = Pi[:, j].sum() + 1e-8
            scalar = d / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * Z_ @ Z_.T)
            compress_loss.append(log_det)
            scalars.append(trPi / (2 * n))
        return compress_loss, scalars

    def deltaR(self, Z, Y, num_classes):
    
        if num_classes is None:
            num_classes = Y.max() + 1
            
        #print("classes:", num_classes)

        Pi = F.one_hot(Y, num_classes).to(Z.device)
        discrimn_loss = self.compute_discrimn_loss(Z.T)
        compress_loss, scalars = self.compute_compress_loss(Z.T, Pi)

        compress_term = 0.
        for z, s in zip(compress_loss, scalars):
            compress_term += s * z
        total_loss = discrimn_loss - compress_term

        return -total_loss, (discrimn_loss, compress_term, compress_loss, scalars)

    def gumb_compress_loss(self, Z, P):
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = 0.
        for j in range(self.num_class):
        
            #P[:, j:j+1][P[:, j:j+1]<threshold] = 0 
            
            Z_ = Z * P[:, j:j+1]
            trPi = P[:, j].sum() + 1e-8
            scalar = d / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * Z_ @ Z_.T)
            compress_loss += (trPi / (2 * n)) *log_det
        return compress_loss

    def pseudo_label_loss(self, Z, logits, thres = 1.4):
    
        logits = logits*thres

        P = F.gumbel_softmax(logits)

        discrimn_loss = self.compute_discrimn_loss(Z.T)
        compress_loss = self.gumb_compress_loss(Z, P)
        total_loss = discrimn_loss - compress_loss

        return -total_loss, (discrimn_loss, compress_loss)