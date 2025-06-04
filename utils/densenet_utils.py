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
    return densenet_models[params['name']](
        spatial_dims=params['spatial_dims'], 
        in_channels=params['n_input_channels'], 
        out_channels=params['num_classes'],
        dropout_prob=params['dropout_prob'],
    )