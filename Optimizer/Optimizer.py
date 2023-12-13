import torch
from torch.optim.optimizer import Optimizer, required

class SGDOptimizer(Optimizer):

    r"""Our Implementation of naive stochastic gradient descent based on pytorch.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        defaults = dict(lr=lr)
        super(SGDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDOptimizer, self).__setstate__(state)
        
    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                p.add_(d_p, alpha=-lr)
        return

class MomentumSGDOptimizer(Optimizer):

    r"""Our Implementation of momentum stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
    """

    def __init__(self, params, lr=required, momentum=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        
        defaults = dict(lr=lr, momentum=momentum)
        super(MomentumSGDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MomentumSGDOptimizer, self).__setstate__(state)
        
    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                p.add_(buf, alpha=-lr)
        return
