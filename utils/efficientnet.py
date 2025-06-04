import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from monai.networks.nets import EfficientNetBN

def get_efficientnet_model(params:dict) -> torch.nn.Module:
    """
    Create EfficientNetBN model with given parameters.
    
    Args:
        params (dict): Dictionary containing model parameters.
        
    Returns:
        torch.nn.Module: EfficientNetBN model instance.
    """
    return EfficientNetBN(
        model_name=params['name'],
        spatial_dims=params['spatial_dims'],
        in_channels=params['n_input_channels'],
        num_classes=params['num_classes']
    )











 