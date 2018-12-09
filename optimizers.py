import torch
import numpy as np

class RMSpropEx(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, beta=0.5, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, alpha=alpha, beta=beta, eps=eps, step=0)
        super(RMSpropEx, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropEx, self).__setstate__(state)
        raise NotImplementedError('not implemented')

    def _get_norm(self, v, norm):
        if norm == 'l2':
            return sum([(p**2).sum() for p in v]).sqrt()
        else:
            raise ValueError('Not supported norm')

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            beta = group['beta']
            eps = group['eps']
            step = group['step']

            group['step'] += 1

            t = float(step + 1)
            squ_bias_correction = 1./(1.-np.power(alpha, t))
            norm_bias_correction = 1./(1.-np.power(beta, t))

            device = None

            updates = []
            update_norm = None
            for p in group['params']:
                if p.grad is None:
                    continue

                if device is None:
                    device = p.device

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSpropEx does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                
                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                avg = square_avg.sqrt().add_(group['eps'])
                avg.mul_(squ_bias_correction)

                update = lr * grad / avg
                updates.append(update)

                if update_norm is None:
                    update_norm = (update ** 2).sum()
                else:
                    update_norm += (update ** 2).sum()

            if 'av_norm' in group.keys():
                av_norm = group['av_norm']
            else:
                av_norm = torch.zeros(1, device=device)
                group['av_norm'] = av_norm

            update_norm = update_norm.sqrt()
            av_norm.mul_(beta).add_((1 - beta)*update_norm)

            update_norm = av_norm*norm_bias_correction + eps

            for p, update in zip(group['params'], updates):
                if p.grad is None:
                    continue

                p.data -= lr * update / update_norm

        return loss
