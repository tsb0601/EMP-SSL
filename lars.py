import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required

class LARS(Optimizer):
    """
    Layer-wise adaptive rate scaling
    - Converted from Tensorflow to Pytorch from:
    https://github.com/google-research/simclr/blob/master/lars_optimizer.py
    - Based on:
    https://github.com/noahgolmant/pytorch-lars
    params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        lr (int): Length / Number of layers we want to apply weight decay, else do not compute
        momentum (float, optional): momentum factor (default: 0.9)
        use_nesterov (bool, optional): flag to use nesterov momentum (default: False)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
            ("\beta")
        eta (float, optional): LARS coefficient (default: 0.001)
    - Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    - Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    """

    def __init__(self, params, lr, len_reduced, momentum=0.9, use_nesterov=False, weight_decay=0.0, classic_momentum=True, eta=0.001):

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            classic_momentum=classic_momentum,
            eta=eta,
            len_reduced=len_reduced
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eta = eta
        self.len_reduced = len_reduced

    def step(self, epoch=None, closure=None):

        loss = None

        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            learning_rate = group['lr']

            # TODO: Hacky
            counter = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: This really hacky way needs to be improved.
                # Note Excluded are passed at the end of the list to are ignored
                if counter < self.len_reduced:
                    grad += self.weight_decay * param

                # Create parameter for the momentum
                if "momentum_var" not in param_state:
                    next_v = param_state["momentum_var"] = torch.zeros_like(
                        p.data
                    )
                else:
                    next_v = param_state["momentum_var"]

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: implementation of layer adaptation
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()

                    trust_ratio = torch.where(w_norm.ge(0), torch.where(
                        g_norm.ge(0), (self.eta * w_norm / g_norm), torch.Tensor([1.0]).to(device)),
                                              torch.Tensor([1.0]).to(device)).item()

                    scaled_lr = learning_rate * trust_ratio
                    
                    grad_scaled = scaled_lr*grad
                    next_v.mul_(momentum).add_(grad_scaled)

                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)

                # Not classic_momentum
                else:

                    next_v.mul_(momentum).add_(grad)

                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (grad)

                    else:
                        update = next_v

                    trust_ratio = 1.0

                    # TODO: implementation of layer adaptation
                    w_norm = torch.norm(param)
                    v_norm = torch.norm(update)

                    device = v_norm.get_device()

                    trust_ratio = torch.where(w_norm.ge(0), torch.where(
                        v_norm.ge(0), (self.eta * w_norm / v_norm), torch.Tensor([1.0]).to(device)),
                                              torch.Tensor([1.0]).to(device)).item()

                    scaled_lr = learning_rate * trust_ratio

                    p.data.add_(-scaled_lr * update)

                counter += 1

        return loss
    
#LARSWrapper from solo-learn repo...
class LARSWrapper:
    def __init__(
        self,
        optimizer: Optimizer,
        eta: float = 1e-3,
        clip: bool = False,
        eps: float = 1e-8,
        exclude_bias_n_norm: bool = False,
    ):
        """Wrapper that adds LARS scheduling to any optimizer.
        This helps stability with huge batch sizes.

        Args:
            optimizer (Optimizer): torch optimizer.
            eta (float, optional): trust coefficient. Defaults to 1e-3.
            clip (bool, optional): clip gradient values. Defaults to False.
            eps (float, optional): adaptive_lr stability coefficient. Defaults to 1e-8.
            exclude_bias_n_norm (bool, optional): exclude bias and normalization layers from lars.
                Defaults to False.
        """

        self.optim = optimizer
        self.eta = eta
        self.eps = eps
        self.clip = clip
        self.exclude_bias_n_norm = exclude_bias_n_norm

        # transfer optim methods
        self.state_dict = self.optim.state_dict
        self.load_state_dict = self.optim.load_state_dict
        self.zero_grad = self.optim.zero_grad
        self.add_param_group = self.optim.add_param_group

        self.__setstate__ = self.optim.__setstate__  # type: ignore
        self.__getstate__ = self.optim.__getstate__  # type: ignore
        self.__repr__ = self.optim.__repr__  # type: ignore

    @property
    def defaults(self):
        return self.optim.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optim.defaults = defaults

    @property  # type: ignore
    def __class__(self):
        return Optimizer

    @property
    def state(self):
        return self.optim.state

    @state.setter
    def state(self, state):
        self.optim.state = state

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    @torch.no_grad()
    def step(self, closure=None):
        weight_decays = []

        for group in self.optim.param_groups:
            weight_decay = group.get("weight_decay", 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group["weight_decay"] = 0

            # update the parameters
            for p in group["params"]:
                if p.grad is not None and (p.ndim != 1 or not self.exclude_bias_n_norm):
                    self.update_p(p, group, weight_decay)

        # update the optimizer
        self.optim.step(closure=closure)

        # return weight decay control to optimizer
        for group_idx, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[group_idx]

    def update_p(self, p, group, weight_decay):
        # calculate new norms
        p_norm = torch.norm(p.data)
        g_norm = torch.norm(p.grad.data)

        if p_norm != 0 and g_norm != 0:
            # calculate new lr
            new_lr = (self.eta * p_norm) / (g_norm + p_norm * weight_decay + self.eps)

            # clip lr
            if self.clip:
                new_lr = min(new_lr / group["lr"], 1)

            # update params with clipped lr
            p.grad.data += weight_decay * p.data
            p.grad.data *= new_lr