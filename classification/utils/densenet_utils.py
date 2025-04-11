import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch

from monai.networks.nets import (
    densenet121, 
    densenet169, 
    densenet201
)

densenet_models = {
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
}

def get_densenet_model(params:dict) -> torch.nn.Module:
    """
    Create DenseNet model with given parameters.
    
    Args:
        params (dict): Dictionary containing model parameters.
        
    Returns:
        torch.nn.Module: DenseNet model instance.
    """
    if params['pretrained']:
        raise NotImplementedError("Pretrained DenseNet models are currently not implemented.")
    
    return densenet_models[params['name']](
        spatial_dims=params['spatial_dims'], 
        in_channels=params['n_input_channels'], 
        out_channels=params['num_classes']
    )