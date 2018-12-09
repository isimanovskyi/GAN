import numpy as np
import torch
import os

def get_axis_same_padding(kernel_size):
    if kernel_size % 2 == 0:
        return kernel_size // 2
    else:
        return (kernel_size - 1) // 2

def get_same_padding(kernel_size):
    p0 = get_axis_same_padding(kernel_size[0])
    p1 = get_axis_same_padding(kernel_size[1])
    return (p0,p1)

class ActivationBlock(torch.nn.Module):
    def __init__(self, act, **kwargs):
        super(ActivationBlock, self).__init__(**kwargs)

        if isinstance(act, str):
            if act == 'relu':
                self.act = torch.nn.Relu()

            elif act == 'LeakyReLU':
                self.act = torch.nn.LeakyReLU(0.2)

            elif act == 'tanh':
                self.act = torch.nn.Tanh()
            # elif act == 'alt':
            #    def f(x):
            #        return tf.where(x < -1., 0.2*x-0.8, tf.where(x > 1., 0.2*x+0.8, x))
            #    return tf.keras.layers.Activation(f)
            else:
                raise ValueError('Unknown activation')
        else:
            self.act = act

    def forward(self, x):
        return self.act(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, activation, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.activation = ActivationBlock(activation)

        self.conj_conv = None
        if in_filters != out_filters:
            self.conj_conv = torch.nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=(1, 1), padding=get_same_padding(kernel_size))

        self.conv1 = torch.nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=(1, 1), padding=get_same_padding(kernel_size))
        self.conv2 = torch.nn.Conv2d(out_filters, out_filters, kernel_size=kernel_size, stride=(1, 1), padding=get_same_padding(kernel_size))

    def forward(self, x):
        if self.conj_conv:
            z = self.conj_conv(x)
        else:
            z = x

        y = self.conv1(x)
        y = self.activation(y)

        y = self.conv2(y)

        x = y + z
        x = self.activation(x)
        return x

class ReshapeBlock(torch.nn.Module):
    def __init__(self, shape, **kwargs):
        super(ReshapeBlock, self).__init__(**kwargs)
        self.shape = shape

    def forward(self, x):
        return x.reshape((-1,) + self.shape)

class FlattenBlock(torch.nn.Module):
    def forward(self, x):
        return x.reshape((x.size(0), -1))

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

    def add_Dense(self, features):
        if len(self.input_shape) != 1:
            raise ValueError('Previous layer is not convolutional')

        #add layer
        self.layers.append(torch.nn.Linear(self.input_shape[0], features))

        #output shape
        self.input_shape = (features,)

    def add_Reshape(self, shape):
        self.layers.append(ReshapeBlock(shape))
        self.input_shape = shape

    def add_Flatten(self):
        self.layers.append(FlattenBlock())
        self.input_shape = (np.prod(self.input_shape),)

    def add_Activation(self, act):
        self.layers.append(ActivationBlock(act))

    def add_Residual(self, filters, kernel_size, activation):
        if len(self.input_shape) != 3:
            raise ValueError('Input is not Convolutional')

        self.layers.append(ResidualBlock(self.input_shape[0],filters,kernel_size,activation))
        self.input_shape = (filters, self.input_shape[1], self.input_shape[2])

    def get(self):
        return SequentialModel(*self.layers)

class ModelBase(object):
    def __init__(self, device, batch, **kwargs):
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

        self.d_model = self._get_discriminator(image_shape = batch.get_image_shape())
        self.d_model.apply(self.init_weights)
        self.d_model.to(device = device)

        self.g_model = self._get_generator(z_shape = batch.get_z_shape(), image_shape = batch.get_image_shape())
        self.g_model.apply(self.init_weights)
        self.g_model.to(device = device)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear \
            or type(m) == torch.nn.Conv2d \
            or type(m) == torch.nn.ConvTranspose2d:
            torch.nn.init.normal_(m.weight, mean=0., std=0.01)
            torch.nn.init.constant_(m.bias, 0.)

    def get_weights(self):
        return self.d_model.parameters(), self.g_model.parameters()

    def _load_weights(self, model, file):
        model.load_state_dict(torch.load(file))

    def _save_weights(self, model, file):
        torch.save(model.state_dict(), file)

    def get_nn_files(self, path):
        d_file = os.path.join(path, 'net_d.npz')
        g_file = os.path.join(path, 'net_g.npz')
        return d_file, g_file

    def save_checkpoint(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        d_file, g_file = self.get_nn_files(path)

        self._save_weights(self.d_model, d_file)
        self._save_weights(self.g_model, g_file)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return False

        d_file, g_file = self.get_nn_files(path)

        try:
            self._load_weights(self.d_model, d_file)
            self._load_weights(self.g_model, g_file)
            return True

        except:
            return False

