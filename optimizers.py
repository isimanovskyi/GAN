import torch
import numpy as np
import logger

class TROptimizer(object):
    def __init__(self, loss, opt, delta=0.2, verbose=False):
        self.delta = delta
        self.epsilon = 1e-12
        self.opt = opt
        self.loss = loss
        self.verbose=verbose

    def zero_grad(self):
        self.z_lst = []
        self.opt.zero_grad()

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)

    def step(self, closure=None):
        with torch.no_grad():
            old_params = {}
            for group in self.opt.param_groups:
                for p in group['params']:
                    old_params[p] = p.clone()

            res = self.opt.step(closure)

            r = self.loss(self.z_lst)

            #get multiplier
            alpha = self.delta / (r + self.epsilon)
            if alpha >= 1:
                return res

            beta = np.clip(alpha, 0., 1.)
            if self.verbose:
                logger.info('TRRmsProp: beta=%.8f, alpha=%.8f, r=%.8f' % (beta, alpha, r))

            # update model
            for group in self.opt.param_groups:
                for p in group['params']:
                    p.copy_(beta * p + (1. - beta) * old_params[p])

            return res

class LrLambdaScheduler(torch.optim.lr_scheduler.LambdaLR):
    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)