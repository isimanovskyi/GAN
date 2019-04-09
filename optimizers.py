import torch
import numpy as np

class Vector(object):
    def __init__(self, v):
        if type(v) is Vector:
            self.v = v.v
        else:
            self.v = [vi for vi in v]

    def __add__(self, v):
        return Vector([v1i + v2i for v1i, v2i in zip(self.v, v.v)])

    def __sub__(self, v):
        return Vector([v1i - v2i for v1i,v2i in zip(self.v, v.v)])

    def __mul__(self, s):
        if type(s) is Vector:
            return Vector([v1i*v2i for v1i,v2i in zip(self.v,s.v)])
        else:
            return Vector([vi * s for vi in self.v])

    def __mod__(self, v):
        return sum([(v1i * v2i).sum() for v1i, v2i in zip(self.v,v.v)])

    def __abs__(self):
        return sum([vi.pow(2).sum() for vi in self.v])

    def clone(self):
        return Vector([vi.clone() for vi in self.v])



class TRRmsProp(object):
    def __init__(self, loss, params, lr, alpha=0.99, delta=0.2, eps=1e-8):
        self.delta = delta
        self.epsilon = eps
        self.opt = torch.optim.RMSprop(params, lr, alpha, eps)
        self.loss = loss

    def zero_grad(self):
        self.z_lst = []
        self.opt.zero_grad()

    def step(self, closure=None):
        with torch.no_grad():
            old_params = {}
            for group in self.opt.param_groups:
                for p in group['params']:
                    old_params[p] = p.clone()

            res = self.opt.step(closure)

            r = self.loss(self.z_lst)

            alpha = self.delta / (r + self.epsilon)
            beta = np.clip(alpha, 0., 1.)
            print (beta, alpha, r)

            # update model
            for group in self.opt.param_groups:
                for p in group['params']:
                    p.copy_(beta * p + (1. - beta) * old_params[p])

            return res

class MSENatGradOptimizer(object):
    def __init__(self, model, params, lr):
        self.model = model
        self.params = [p for p in params]
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def _get_grad(self):
        return Vector([p.grad.clone() for p in self.params])

    def _get_hessian_vector_product(self, z_list, v):
        self.zero_grad()

        device = next(self.model.parameters()).device

        R = 0.
        for z, samples in z_list:
            gen_samples = self.model(z.to(device))
            with torch.no_grad():
                samples_ng = gen_samples.clone().detach()
            R += (gen_samples - samples_ng).pow(2).mean()

        grad_f = torch.autograd.grad(R, self.params, create_graph=True)

        ip = Vector(grad_f) % v
        ip.backward()

        res = self._get_grad()
        return res

    def _conjugate_grad(self, g, z_list, max_iter = 10):
        x = g
        r = g - self._get_hessian_vector_product(z_list, x)
        p = r.clone()
        r_norm = abs(r)

        for i in xrange(max_iter):
            Hp = self._get_hessian_vector_product(z_list, p)
            alpha = r_norm / (p % Hp)

            x = x + p * alpha
            r = r - Hp * alpha

            #Hx = self._get_hessian_vector_product(z_list, x)
            #J = 0.5*(x%Hx) - g%x
            #print (J)

            r_norm_1 = abs(r)

            if r_norm_1 < 1e-3:
                print ('it %d'%(i))
                break

            p = r + p * (r_norm_1/r_norm)
            r_norm = r_norm_1

        return x

    def step(self, z_list):
        g = self._get_grad()
        g = self._conjugate_grad(g, z_list)


