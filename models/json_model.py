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
        self.variables = data['variables']

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

        args = self.arg_hook(args)

        getattr(net, method_name)(**args)

        if 'use_batch_norm' in layer.keys():
            use_bn = self.convert_value(layer['use_batch_norm'])
            if use_bn:
                net.add_BatchNorm()

        if 'activation' in layer.keys():
            activation = self.convert_value(layer['activation'])
            net.add_Activation(activation)

    def convert_value(self, val):
        if type(val) is not str and type(val) is not unicode:
            return val

        if not val.startswith('var/'):
            return val

        var_name = val.split('var/', 1)[1]
        return self.variables[var_name]

    def arg_hook(self, args):
        for key, value in args.iteritems():
            args[key] = self.convert_value(value)
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
        