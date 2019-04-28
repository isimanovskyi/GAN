import numpy as np
import torch
import os
import sys, inspect
import functools
import utils
import copy

def get_axis_same_padding(kernel_size):
    if kernel_size % 2 == 0:
        return kernel_size // 2
    else:
        return (kernel_size - 1) // 2

def get_same_padding(kernel_size):
    p0 = get_axis_same_padding(kernel_size[0])
    p1 = get_axis_same_padding(kernel_size[1])
    return (p0,p1)

def convert_padding(padding, kernel_size):
    if utils.is_str(padding):
        if padding == 'same':
            return get_same_padding(kernel_size)
        elif padding == 'valid':
            return (0, 0)
        else:
            ValueError('Unknown padding type')

    return tuple(padding)

class IdentityBlock(torch.nn.Module):
    def __init__(self, **kwargs):
        super(IdentityBlock, self).__init__(**kwargs)

    def forward(self, x):
        return x

class ActivationBlock(torch.nn.Module):
    def __init__(self, act, **kwargs):
        super(ActivationBlock, self).__init__(**kwargs)

        if act is None:
            raise ValueError('Activation cannot be None')

        if isinstance(act, str):
            if act == 'relu':
                self.act = torch.nn.Relu()

            elif act == 'LeakyReLU':
                self.act = torch.nn.LeakyReLU(0.2)

            elif act == 'tanh':
                self.act = torch.nn.Tanh()
            else:
                raise ValueError('Unknown activation')

        if isinstance(act, dict):
            for name, cs in inspect.getmembers(sys.modules['torch.nn'], inspect.isclass):
                if name == act['type']:
                    self.act = cs(**act['args'])
                    return
                    
            raise ValueError('Unknown object')

        else:
            self.act = act

    def forward(self, x):
        return self.act(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, activation, use_batch_norm, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.use_batch_norm = use_batch_norm
        self.activation = ActivationBlock(activation)

        self.conj_conv = None
        if in_filters != out_filters:
            self.conj_conv = torch.nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=(1, 1), padding=get_same_padding(kernel_size))
            if use_batch_norm:
                self.conj_bn = torch.nn.BatchNorm2d(out_filters)

        self.conv1 = torch.nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=(1, 1), padding=get_same_padding(kernel_size))
        if use_batch_norm:
            self.conv1_bn = torch.nn.BatchNorm2d(out_filters)

        self.conv2 = torch.nn.Conv2d(out_filters, out_filters, kernel_size=kernel_size, stride=(1, 1), padding=get_same_padding(kernel_size))
        if use_batch_norm:
            self.conv2_bn = torch.nn.BatchNorm2d(out_filters)

    def forward(self, x):
        if self.conj_conv:
            z = self.conj_conv(x)
            if self.use_batch_norm:
                z = self.conj_bn(z)
        else:
            z = x

        y = self.conv1(x)
        if self.use_batch_norm:
            y = self.conv1_bn(y)
        y = self.activation(y)

        y = self.conv2(y)
        if self.use_batch_norm:
            y = self.conv2_bn(y)

        return self.activation(y + z)

class ResidualBlock2(torch.nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, activation, use_batch_norm, **kwargs):
        super(ResidualBlock2, self).__init__(**kwargs)
        self.resblock = ResidualBlock(in_filters, out_filters, kernel_size, activation, use_batch_norm, **kwargs)

    def forward(self, x):
        if self.resblock.conj_conv:
            return self.resblock(x)
        return x + self.resblock(x)

class ReshapeBlock(torch.nn.Module):
    def __init__(self, shape, keep_batch=True,**kwargs):
        super(ReshapeBlock, self).__init__(**kwargs)
        self.shape = tuple(shape)
        self.keep_batch = keep_batch

    def forward(self, x):
        if self.keep_batch:
            return x.reshape((x.size(0),) + self.shape)
        return x.reshape(self.shape)

class FlattenBlock(torch.nn.Module):
    def forward(self, x):
        return x.reshape((x.size(0), -1))


class SelfAttentionBlock(torch.nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation = None):
        super(SelfAttentionBlock, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        self.softmax = torch.nn.Softmax(dim=-1)  #

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height)
        if self.activation is not None:
            proj_query = self.activation(proj_query)

        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        if self.activation is not None:
            proj_key = self.activation(proj_key)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        if self.activation is not None:
            proj_value = self.activation(proj_value)

        out = torch.bmm(proj_value, proj_key.permute(0,2,1))
        out = torch.bmm(out, proj_query)

        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class NormBlock(torch.nn.Module):
    def __init__(self, eps=1e-7, **kwargs):
        super(NormBlock, self).__init__(**kwargs)
        self.epsilon = eps

    def forward(self, x):
        return (x - x.mean(0))/(x.std(0) + self.epsilon)

class SequentialModel(torch.nn.Sequential):
    def __init__(self, *kwargs):
        super(SequentialModel, self).__init__(*kwargs)

    def requires_grad(self, req_grad):
        for p in self.parameters():
            p.requires_grad = req_grad

class SequentialContainer(object):
    def __init__(self, input_shape):
        self.layers = []
        self.input_shape = input_shape

    def add_Conv2D(self, channels, kernel_size, strides=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=True):
        if groups != 1:
            raise NotImplementedError('groups must be 1')

        if len(self.input_shape) != 3:
            raise ValueError('Input is not Convolutional')

        padding = convert_padding(padding, kernel_size)

        #add layer
        self.layers.append(torch.nn.Conv2d(self.input_shape[0], channels, kernel_size, strides, padding, dilation, groups, bias))
        
        #output shape
        h, w = self.input_shape[1], self.input_shape[2]
        ker_h, ker_w = kernel_size
        str_h, str_w = strides
        pad_h, pad_w = padding
        dil_h, dil_w = dilation

        h = int((h + 2*pad_h - dil_h *(ker_h-1) - 1)/str_h + 1)
        w = int((w + 2*pad_w - dil_w *(ker_w-1) - 1)/str_w + 1)

        self.input_shape = (channels, h, w)

    def add_Conv2DTranspose(self, channels, kernel_size, strides=(1,1), padding=(0,0), output_padding=(0,0), groups=1, bias=True, dilation=(1,1)):
        if groups != 1:
            raise NotImplementedError('groups must be 1')

        if len(self.input_shape) != 3:
            raise ValueError('Input is not Convolutional')

        padding = convert_padding(padding, kernel_size)

        #add layer
        self.layers.append(torch.nn.ConvTranspose2d(self.input_shape[0], channels, kernel_size, strides, padding, output_padding, groups, bias, dilation))
        
        #output shape
        h, w = self.input_shape[1], self.input_shape[2]
        ker_h, ker_w = kernel_size
        str_h, str_w = strides
        pad_h, pad_w = padding
        opad_h, opad_w = output_padding
        dil_h, dil_w = dilation

        h = (h-1)*str_h - 2*pad_h + ker_h + opad_h
        w = (w-1)*str_w - 2*pad_w + ker_w + opad_w

        self.input_shape = (channels, h, w)

    def add_Dense(self, features, bias=True):
        if len(self.input_shape) != 1:
            raise ValueError('Input is not flat')

        #add layer
        self.layers.append(torch.nn.Linear(self.input_shape[0], features, bias))

        #output shape
        self.input_shape = (features,)

    def add_Reshape(self, shape, keep_batch = True):
        self.layers.append(ReshapeBlock(shape, keep_batch))
        if keep_batch:
            self.input_shape = shape
        else:
            self.input_shape = tuple(shape[1:])

    def add_Flatten(self):
        self.layers.append(FlattenBlock())
        self.input_shape = (np.prod(self.input_shape),)

    def add_Activation(self, activation):
        self.layers.append(ActivationBlock(activation))

    def add_Residual(self, filters, kernel_size, activation, use_batch_norm = False):
        if len(self.input_shape) != 3:
            raise ValueError('Input is not Convolutional')

        self.layers.append(ResidualBlock2(self.input_shape[0],filters,kernel_size,activation, use_batch_norm))
        self.input_shape = (filters, self.input_shape[1], self.input_shape[2])

    def add_BatchNorm(self, affine=True):
        if len(self.input_shape) == 1:
            self.layers.append(torch.nn.BatchNorm1d(self.input_shape[0], affine=affine))
        elif len(self.input_shape) == 3:
            self.layers.append(torch.nn.BatchNorm2d(self.input_shape[0], affine=affine))
        else:
            raise ValueError('Unsupported shape')

    def add_NormBlock(self):
        self.layers.append(NormBlock())

    def add_SelfAttention(self, activation=None):
        if len(self.input_shape) != 3:
            raise ValueError('Input is not Convolutional')

        if activation is not None:
            activation = ActivationBlock(activation)

        self.layers.append(SelfAttentionBlock(self.input_shape[0], activation))

    def add_AvgPooling(self):
        if len(self.input_shape) != 3:
            raise ValueError('Input is not Convolutional')

        self.layers.append(torch.nn.AdaptiveAvgPool2d((1,1)))
        self.input_shape = (self.input_shape[0], 1, 1)

    def add_Identity(self):
        self.layers.append(IdentityBlock())

    def add_Upsample2d(self,scale_factor):
        if len(self.input_shape) != 3:
            raise ValueError('Input is not Convolutional')

        self.layers.append(torch.nn.UpsamplingNearest2d(size=None, scale_factor=scale_factor))
        self.input_shape = (self.input_shape[0], scale_factor*self.input_shape[1], scale_factor*self.input_shape[2])

    def get(self):
        return SequentialModel(*self.layers)

class Discriminator(torch.nn.Module):
    def __init__(self, features, fc, ae = None, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.features = features
        self.fc = fc
        self.ae = ae

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

    def requires_grad(self, req_grad):
        for p in self.parameters():
            p.requires_grad = req_grad

class ModelBase(object):
    def __init__(self, device, z_shape, image_shape, use_av_gen = False, **kwargs):
        self.device = device

        if 'g_tanh' in kwargs.keys():
            self.g_tanh = kwargs['g_tanh']
        else:
            self.g_tanh = False

        if 'g_act' in kwargs.keys():
            self.g_act = kwargs['g_act']
        else:
            self.g_act = torch.nn.LeakyReLU(0.2)

        if 'd_act' in kwargs.keys():
            self.d_act = kwargs['d_act']
        else:
            self.d_act = torch.nn.LeakyReLU(0.2)

        self.d_model = self._get_discriminator(image_shape=image_shape)
        self.d_model.apply(functools.partial(self.init_weights, sigma=0.005))
        self.d_model.to(device = device)

        self.g_model = self._get_generator(z_shape=z_shape, image_shape=image_shape)
        self.g_model.apply(functools.partial(self.init_weights, sigma=0.01))
        self.g_model.to(device = device)

        if use_av_gen:
            self.av_g_model = copy.deepcopy(self.g_model)
            self.av_g_model.requires_grad(False)

    def init_weights(self, m, sigma):
        if isinstance(m, torch.nn.Linear) \
            or isinstance(m, torch.nn.Conv2d) \
            or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, mean=0., std=sigma)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.)

    def get_weights(self):
        return self.d_model.parameters(), self.g_model.parameters()

    def update_g_av(self, beta = 0.999):
        if not hasattr(self, 'av_g_model'):
            return

        param_dict_src = dict(self.g_model.named_parameters())

        for p_name, p_tgt in self.av_g_model.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    def _load_weights(self, model, file):
        model.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))

    def _save_weights(self, model, file):
        torch.save(model.state_dict(), file)

    def get_nn_files(self, path):
        d_file = os.path.join(path, 'net_d.npz')
        g_file = os.path.join(path, 'net_g.npz')
        g_av_file = os.path.join(path, 'net_g_av.npz')
        return d_file, g_file, g_av_file

    def save_checkpoint(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        d_file, g_file, g_av_file = self.get_nn_files(path)

        self._save_weights(self.d_model, d_file)
        self._save_weights(self.g_model, g_file)
        if hasattr(self, 'av_g_model'):
            self._save_weights(self.av_g_model, g_av_file)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return False

        d_file, g_file, g_av_file = self.get_nn_files(path)

        try:
            self._load_weights(self.d_model, d_file)
            self._load_weights(self.g_model, g_file)

            try:
                if hasattr(self, 'av_g_model'):
                    self._load_weights(self.av_g_model, g_av_file)
            except Exception as e:
                self.av_g_model = copy.deepcopy(self.g_model)
                self.av_g_model.requires_grad(False)
            return True

        except Exception as e:
            print (e)
            return False

    def train(self):
        self.d_model.train()
        self.g_model.train()

