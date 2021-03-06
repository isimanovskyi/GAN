import models.model
import models.json_model
import os
import sys, inspect

def create_model(name, **kwargs):
    if name.startswith('json_'):
        model_name = name.split('json_', 1)[1]
        d_file = os.path.join('json_models', model_name, 'discriminator.json')
        g_file = os.path.join('json_models', model_name, 'generator.json')
        return models.json_model.JSONModel(d_file=d_file, g_file=g_file, **kwargs)

    #find model from model.py
    for n, cs in inspect.getmembers(sys.modules['models.model'], inspect.isclass):
        if n == name:
            return cs(**kwargs)

    raise ValueError('Unknown model')



