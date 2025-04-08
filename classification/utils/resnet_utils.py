import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from monai.networks.nets import (
    resnet18, 
    resnet34, 
    resnet50, 
    resnet101
)


def get_pretrained_resnet18(spatial_dims:int, n_input_channels:int, num_classes:int) -> torch.nn.Module:
    return resnet18(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=num_classes, 
        pretrained=True, 
        feed_forward=False, 
        shortcut_type='B', 
        bias_downsample=False
    )


def get_pretrained_resnet34(spatial_dims:int, n_input_channels:int, num_classes:int) -> torch.nn.Module:
    return resnet34(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=num_classes, 
        pretrained=True, 
        feed_forward=False, 
        shortcut_type='B', 
        bias_downsample=False
    )


def get_pretrained_resnet50(spatial_dims:int, n_input_channels:int, num_classes:int) -> torch.nn.Module:
    return resnet50(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=num_classes, 
        pretrained=True, 
        feed_forward=False, 
        shortcut_type='B', 
        bias_downsample=False
    )


def get_pretrained_resnet101(spatial_dims:int, n_input_channels:int, num_classes:int) -> torch.nn.Module:
    return resnet101(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=num_classes, 
        pretrained=True, 
        feed_forward=False, 
        shortcut_type='B', 
        bias_downsample=False
    )


resnet18p = get_pretrained_resnet18
resnet34p = get_pretrained_resnet34
resnet50p = get_pretrained_resnet50
resnet101p = get_pretrained_resnet101

resnet_models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet18p': resnet18p,
    'resnet34p': resnet34p,
    'resnet50p': resnet50p,
    'resnet101p': resnet101p
}


def get_resnet_model(params:dict) -> torch.nn.Module:
    """
    Create ResNet model with given parameters.
    
    Args:
        params (dict): Dictionary containing model parameters.
        
    Returns:
        torch.nn.Module: ResNet model instance.
    """
    return resnet_models[params['name']](
        spatial_dims=params['spatial_dims'], 
        n_input_channels=params['n_input_channels'], 
        num_classes=params['num_classes']
    )