import numpy as np
import torch
import models.base
import json
import logger

class Net(object):
    def __init__(self, file):
        self.file = file

    def parse(self, in_shape):
        data = None
        with open(self.file) as f:
            data = json.load(f)

        #fill model ctx
        self.name = data['name']

        if 'activation' in data.keys():
            self.activation = data['activation']
        else:
            self.activation = None

        if 'use_batch_norm' in data.keys():
            self.use_batch_norm = data['use_batch_norm']
        else:
            self.use_batch_norm = False

        #create model
        net = models.base.SequentialContainer(in_shape)

        for layer in data['layers']:
            self.parse_layer(net, layer)

        return net.get()

    def parse_layer(self, net, layer):
        method_name = 'add_' + layer['type']

        if 'args' in layer.keys():
            args = layer['args']
        else:
            args = {}

        args = self.arg_hook(layer, args)

        getattr(net, method_name)(**args)

        if 'use_batch_norm' in layer.keys():
            if layer['use_batch_norm']:
                net.add_BatchNorm()
        elif self.use_batch_norm and layer['type'] != 'Residual':
            net.add_BatchNorm()

        if 'activation' in layer.keys():
            activation = layer['activation']
            if type(activation) is dict and activation['type'] == 'Parent':
                net.add_Activation(self.activation)
            else:
                net.add_Activation(activation)

    def arg_hook(self, layer, args):
        if 'activation' in args.keys():
            activation = args['activation']
            if type(activation) is dict and activation['type'] == 'Parent':
                args['activation'] = self.activation

        if layer['type'] == 'Residual':
            if 'use_batch_norm' not in args.keys():
                args['use_batch_norm'] = self.use_batch_norm

        return args

class JSONModel(models.base.ModelBase):
    def __init__(self, d_file, g_file, **kwargs):
        self.d_file = d_file
        self.g_file = g_file

        super(JSONModel, self).__init__(**kwargs)

    def _get_generator(self, z_shape, image_shape):
        return Net(self.g_file).parse(z_shape)
 
    def _get_discriminator(self, image_shape):
        return Net(self.d_file).parse(image_shape)
        