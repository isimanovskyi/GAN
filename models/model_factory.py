import models.model
import models.json_model
import os

def create_model(name, **kwargs):
    if name == 'Model':
         return models.model.Model(**kwargs)
    elif name == 'ResidualModel':
        return models.model.Model(**kwargs)
    elif name == 'DeepResidualModel':
        return models.model.DeepResidualModel(**kwargs)
    elif name.startswith('json_'):
        model_name = name.split('json_', 1)[1]
        d_file = os.path.join('json_models', model_name, 'discriminator.json')
        g_file = os.path.join('json_models', model_name, 'generator.json')
        return models.json_model.JSONModel(d_file=d_file, g_file=g_file, **kwargs)

    raise ValueError('Unknown model')



