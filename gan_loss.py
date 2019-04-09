import numpy as np
import torch
import mmd.kernels

class loss_base(object):
    def __init__(self, **kwargs):
        if 'eps' in kwargs.keys():
            self.eps = kwargs['eps']
        else:
            self.eps = 1e-7

    def fd(self, x):
        raise NotImplemented('This is abstract method')

    def fg(self, x):
        raise NotImplemented('This is abstract method')
    
    def fg2(self, x):
        raise NotImplemented('This is abstract method')
        
    def get_d_real(self, real_d):
        return self.fd(real_d).mean()

    def get_d_fake(self, fake_d):
        return self.fg(fake_d).mean()

    def get_g(self, fake_d):
        return self.fg2(fake_d).mean()

    @property
    def zero_level(self):
        raise NotImplemented('This is abstract method')

class pearson_loss(loss_base):
    def __init__(self, **kwargs):
        super(pearson_loss, self).__init__(**kwargs)
        
    def fd(self, x):
        return (x**2)/2

    def fg(self, x):
        return -x
    
    def fg2(self, x):
        return x

    @property
    def zero_level(self):
        return -0.5

class crossentropy_loss(loss_base):
    def __init__(self, **kwargs):
        super(crossentropy_loss, self).__init__(**kwargs)

    def fd(self, x):
        return torch.nn.functional.binary_cross_entropy_with_logits(x, torch.ones_like(x), reduction='none')

    def fg(self, x):
        return torch.nn.functional.binary_cross_entropy_with_logits(x, torch.zeros_like(x), reduction='none')

    @property
    def zero_level(self):
        return 1.3863
    
class js_loss(crossentropy_loss):
    def __init__(self, **kwargs):
        super(js_loss, self).__init__(**kwargs)
        
    def fg2(self, x):
        return -torch.nn.functional.binary_cross_entropy_with_logits(x, torch.zeros_like(x), reduction='none')
    
class goodfellow_loss(crossentropy_loss):
    def __init__(self, **kwargs):
        super(goodfellow_loss, self).__init__(**kwargs)
    
    def fg2(self, x):
        return torch.nn.functional.binary_cross_entropy_with_logits(x, torch.ones_like(x), reduction='none')
    
class huzcar_loss(crossentropy_loss):
    def __init__(self, **kwargs):
        super(huzcar_loss, self).__init__(**kwargs)
        
    def fg2(self, x):
        return -x

class wasserstein_loss(loss_base):
    def __init__(self, **kwargs):
        super(wasserstein_loss, self).__init__(**kwargs)
        
    def fd(self, x):
        return -x

    def fg(self, x):
        return x
    
    def fg2(self, x):
        return -x

    @property
    def zero_level(self):
        return 0.

class mmd_loss(loss_base):
    def __init__(self, **kwargs):
        super(mmd_loss, self).__init__(**kwargs)
        self.allows_separate_d = False
        self.kernel = mmd.kernels.GaussKernel([0.01, 0.1, 1.,2.,10.,30, 100., 1000])

    def get_d(self, real_d, fake_d):
        return -self.kernel.get_linear_score(real_d, fake_d)

    def get_g(self, real_d, fake_d):
        return self.kernel.get_linear_score(real_d, fake_d)
