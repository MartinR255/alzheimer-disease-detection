import os
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
from monai.networks.nets import (
    resnet18, 
    resnet34, 
    resnet50, 
    resnet101
)


def get_pretrained_resnet18(spatial_dims:int, n_input_channels:int, num_classes:int) -> torch.nn.Module:
    model = resnet18(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=num_classes, 
        pretrained=True, 
        feed_forward=False, 
        shortcut_type='A', 
        bias_downsample=True
    )
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return model


def get_pretrained_resnet34(spatial_dims:int, n_input_channels:int, num_classes:int) -> torch.nn.Module:
    model = resnet34(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=num_classes, 
        pretrained=True, 
        feed_forward=False, 
        shortcut_type='A', 
        bias_downsample=True
    )
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return model


def get_pretrained_resnet50(spatial_dims:int, n_input_channels:int, num_classes:int) -> torch.nn.Module:
    model = resnet50(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=num_classes, 
        pretrained=True, 
        feed_forward=False, 
        shortcut_type='B', 
        bias_downsample=False
    )
    model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    return model


def get_pretrained_resnet101(spatial_dims:int, n_input_channels:int, num_classes:int) -> torch.nn.Module:
    model = resnet101(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=num_classes, 
        pretrained=True, 
        feed_forward=False, 
        shortcut_type='B', 
        bias_downsample=False
    )
    model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    return model


def add_dropout_relu(model:torch.nn.Module, dropout_rate:float) -> torch.nn.Module:
    """
    Recursively loops model and add dropout after ReLU activation.

    Solution from: https://discuss.pytorch.org/t/where-and-how-to-add-dropout-in-resnet18/12869/2
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            add_dropout_relu(module, dropout_rate)
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout(p=dropout_rate, inplace=False))
            setattr(model, name, new)


def add_dropout_fc(model:torch.nn.Module, dropout_rate:float) -> None:
    new = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        model.fc
    )
    setattr(model, 'fc', new)


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
    model = resnet_models[params['name']](
        spatial_dims=params['spatial_dims'], 
        n_input_channels=params['n_input_channels'], 
        num_classes=params['num_classes']
    )
    
    if pd.isna(params['dropout_rate_relu']) is False:
        add_dropout_relu(model, params['dropout_rate_relu'])

    if pd.isna(params['dropout_rate_fc']) is False:
        add_dropout_fc(model, params['dropout_rate_fc'])

    return model