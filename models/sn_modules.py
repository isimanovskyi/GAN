import torch
import torch.nn.functional as F

def power_it(module, u, max_it, eps = 1e-12):
    _u = u
    for it in range(max_it):
        _v = module._dot(_u)
        _v_norm = _v.pow(2).sum().sqrt()
        _v = _v / (_v_norm + eps)
            
        _u = module._dot_t(_v)
        _u_norm = _u.pow(2).sum().sqrt()
        _u = _u / (_u_norm + eps)
        
    u = _u.clone().detach_()
    sigma = (module._dot_t(_v) * _u).sum()
    return sigma, u
    
def get_sn(module, x, max_it = 1, eps = 1e-12):
    if max_it < 1:
        raise ValueError("The number of power iterations should be positive integer")
        
    if not hasattr(module, 'u'):
        with torch.no_grad():
            u = torch.rand((1,) + x.shape[1:], device=x.device)
            sigma, module.u = power_it(module, u, max(10, max_it), eps)
            
    sigma, module.u = power_it(module, module.u, max_it, eps)
    return sigma

def normalize_weights(w , alpha, sigma, lambd):
    return w / ((1. + lambd*alpha*sigma) / (1. + lambd))

class SNConv2d(torch.nn.Conv2d):
    def _dot(self, x):
        return F.conv2d(x,
                        weight=self.weight,
                        bias = None,
                        stride=self.stride,
                        padding=0,#self.padding,
                        dilation=self.dilation,
                        groups=self.groups)
    
    def _dot_t(self, x):
        return F.conv_transpose2d(x,
                                  weight=self.weight,
                                  bias=None,
                                  stride=self.stride,
                                  padding=0,
                                  output_padding=0,#padding,
                                  groups=self.groups,
                                  dilation=self.dilation)

    def get_sn(self):
        return get_sn(self, None, 1)
    
    def forward(self, x):
        if not hasattr(self, 'u'):
            sigma = get_sn(self, x, 1)

        return super(SNConv2d, self).forward(x)

        #sigma = get_sn(self, x, 1)
        #w = normalize_weights(self.weight.data, 1., sigma, self.lambd)
        #r = F.conv2d(x, w, self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        #return r
    
class SNLinear(torch.nn.Linear):
    def _dot(self, x):
        return F.linear(x, self.weight, None)
    
    def _dot_t(self, x):
        return x.matmul(self.weight)

    def get_sn(self):
        return get_sn(self, None, 1)
    
    def forward(self, x):
        if not hasattr(self, 'u'):
            sigma = get_sn(self, x, 1)

        return super(SNLinear, self).forward(x)
    
    #def forward(self, x):
    #    sigma = get_sn(self, x, 1)
    #    w = normalize_weights(self.weight.data, 1., sigma, self.lambd)
    #    return F.linear(x, w, self.bias)