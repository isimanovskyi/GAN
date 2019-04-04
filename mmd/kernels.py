import numpy as np
import torch

class Kernel(object):
    def __call__(self, z):
        raise NotImplementedError('This is abstract class')
        
    def compute(self, x, y):
        x = x.view((x.shape[0], -1))
        y = y.view((y.shape[0], -1))

        xx = torch.mm(x, x.t())
        xy = torch.mm(x, y.t())
        yy = torch.mm(y, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        k_xx = self.__call__(rx.t() + rx - 2*xx)
        k_xy = self.__call__(rx.t() + ry - 2*xy)
        k_yy = self.__call__(ry.t() + ry - 2*yy)

        return k_xx, k_xy, k_yy
        
    def get_score(self, x, y):
        k_xx, k_xy, k_yy = self.compute(x, y)

        m = k_xx.shape[0]
        n = k_yy.shape[0]

        mmd = k_xx.sum() / (m * (m - 1)) + k_yy.sum() / (n * (n - 1)) - 2 * k_xy.sum() / (m * n)
        return mmd
    
    def get_linear_score(self, x, y):
        axis = list(range(len(x.shape)))
        axis = tuple(axis[1:])
        n = x.shape[0]
        m = n // 2
        
        kxx = (x[0:m] - x[m:2*m])**2
        if len(axis) > 0:
            kxx = torch.sum(kxx, axis)
        kxx = self.__call__(kxx)

        kyy = (y[0:m] - y[m:2*m])**2
        if len(axis) > 0:
            kyy = torch.sum(kyy, axis)
        kyy = self.__call__(kyy)
        
        kxy = (x[0:m] - y[m:2*m])**2
        if len(axis) > 0:
            kxy = torch.sum(kxy, axis)
        kxy = self.__call__(kxy)
        
        kyx = (y[0:m] - x[m:2*m])**2
        if len(axis) > 0:
            kyx = torch.sum(kyx, axis)
        kyx = self.__call__(kyx)

        return torch.mean(kxx + kyy - kxy - kyx)
    
    def get_witness(self, x, y, points):
        _, kxp, _ = self.compute(x, points, False)
        _, kyp, _ = self.compute(y, points, False)

        kx = torch.sum(kxp, 0)
        ky = torch.sum(kyp, 0)

        n = points.shape[0]
        return (kx - ky)/n
            
class GaussKernel(Kernel):
    def __init__(self, sigma, epsilon = 1e-12):
        self.epsilon = epsilon
        try:
            self.sigma = [s for s in sigma]
        except TypeError:
            self.sigma = [sigma]

        
    def __call__(self, z):
        z = torch.max(z, torch.zeros_like(z))
        r = 0.
        for s in self.sigma:
            r += torch.exp(-z / (2. * (s**2) + self.epsilon))
        return r / float(len(self.sigma))

class PlummerKernel(Kernel):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        
    def __call__(self, z):
        z = torch.max(z, torch.zeros_like(z))
        return 1./torch.sqrt(z + self.epsilon**2)
