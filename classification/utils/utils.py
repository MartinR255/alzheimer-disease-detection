import os
import yaml
import json
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch

from torch.nn import Parameter
from memory_dataset import MemoryDataset
from monai.data import ImageDataset
from monai.losses import FocalLoss

from typing import Iterable

from .intensity_normalization import IntensityNormalization
from monai.transforms import (
    Compose, 
    LoadImage, 
    Resize, 
    EnsureChannelFirst, 
    CropForeground,
    ToTensor,
    GaussianSmooth
)

from sklearn.model_selection import train_test_split
from .resnet_utils import get_resnet_model
from .densenet_utils import get_densenet_model
from .efficientnet import get_efficientnet_model
from .ad3dcnn import get_custom_net



__all__ = [
    'make_file_dir',
    'copy_config',
    'load_data',
    'get_transform',
    'get_memory_dataset',   
    'get_image_dataset',
    'save_dataset_to_file',
    'stratified_split',
    'load_yaml_config',
    'get_optimizer',
    'get_loss',
    'get_network',
    'get_scheduler'
]

def make_file_dir(file_path:str):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def copy_config(file_path:str, new_file_path:str):
    make_file_dir(new_file_path)

    data = load_yaml_config(file_path)
    with open(new_file_path, 'w') as file:
        yaml.dump(data, file)


def load_data(image_dataset_path: str, dataset_partiton_path: str) -> tuple:
    """
    Load dataset from a JSON file.
    
    Args:
        image_dataset_path (str): Path to the directory containing image files.
        dataset_partiton_path (str): Path to the JSON file containing image filenames and labels.
    
    Returns:
        tuple: List of dictionaries, each containing 'image' path and 'label' for a sample.
    """
    images = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int64)
    with open(dataset_partiton_path, 'r') as data_file:
        data = json.load(data_file)
        for image, label in data.items():
            images = np.append(images, os.sep.join([image_dataset_path, image]))
            labels = np.append(labels, label)
    return images, labels



def get_transform() -> Compose:
    """
    Creates a transformation pipeline for image preprocessing before feeding to the model.
    """
    def select_fn(x):
        return x > 0.00001
    
    data_transform = [
            LoadImage(reader="monai.data.ITKReader"),
            EnsureChannelFirst(),
            CropForeground(
                select_fn=select_fn,
                allow_smaller=False,
                margin=0,
            ),
            GaussianSmooth(sigma=1),
            IntensityNormalization(clip_ratio=99.5),
            Resize(spatial_size=(128, 128, 128), mode='trilinear'),
            ToTensor(dtype=torch.float16)
    ]

    return Compose(data_transform)


def get_memory_dataset(dataset_path:str = None) -> MemoryDataset:
    """
    Creates a MemoryDataset from file.
    """
    return MemoryDataset(path=dataset_path)


def get_image_dataset(image_dataset_path:str, dataset_partiton_path:str) -> ImageDataset:
    """
    Creates a ImageDataset and adds transformations to images.
    """
    images, labels = load_data(image_dataset_path, dataset_partiton_path)
    transform = get_transform()
    dataset = ImageDataset(
        image_files=images,
        labels=labels,
        transform=transform
    )
    return dataset


def save_dataset_to_file(images_path:str, partition_path:str, save_path:str):
    """
    Saves image data and labels as tensors to one file

    Args: 
        images_path (str): Path to the directory containing image files.
        partition_path (str): Path to the JSON file containing image filenames and labels.
        save_path (str): Path where the dataset will be saved.

    Returns:
        None
    """
    images, labels = load_data(images_path, partition_path)
    if len(images) != len(labels):
        raise ValueError(f"Number of images ({len(images)}) does not match number of labels ({len(labels)}).")
    
    transform = get_transform()
    labels = torch.tensor(labels, dtype=torch.int64)
    image_buffer = torch.empty([len(images), 1, 128, 128, 128], dtype=torch.float16) 
    for i, img_path in enumerate(images):
        img_tensor =  transform(img_path)
        image_buffer[i] = img_tensor.to(dtype=torch.float16)
        
    torch.save({"images" : image_buffer, "labels" : labels}, save_path)


def stratified_split(images, labels, ratios:tuple = (0.8, 0.1, 0.1), seed:int = 42) -> tuple:
    """
    Split a dataset into train, validation and test sets using stratified sampling.
    
    Args:
        images (list or array): List of image data or paths.
        labels (list or array): List of corresponding labels.
        ratios (tuple, optional): Proportions for train, validation, and test sets. 
        seed (int, optional): Random seed for reproducibility. 
    
    Returns:
        tuple: Three tuples, each containing:
            - train_ds ([list, list]): Lists of training images and labels.
            - val_ds ([list, list]): Lists of validation images and labels.
            - test_ds ([list, list]): Lists of test images and labels.
    """
    train_size, val_size, test_size = ratios

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, 
        labels, 
        test_size=test_size, 
        stratify=labels, 
        random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, 
        y_train_val, 
        test_size=val_size/(train_size + val_size), 
        stratify=y_train_val, 
        random_state=seed
    )

    train_ds =  [X_train, y_train]
    val_ds =  [X_val, y_val]
    test_ds =  [X_test, y_test]
    return train_ds, val_ds, test_ds


def load_yaml_config(file_path:str) -> dict:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


"""
Optimizer functions
"""
def get_adam_optimizer(model_params:Iterable[Parameter], params:dict) -> torch.optim.Adam:
    return torch.optim.Adam(
        model_params, 
        lr=params['lr'], 
        betas=params['betas'], 
        weight_decay=params['weight_decay']
    )


def get_adamw_optimizer(model_params:Iterable[Parameter], params:dict) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model_params, 
        lr=params['lr'], 
        betas=params['betas'], 
        weight_decay=params['weight_decay']
    )


def get_rmsprop_optimizer(model_params:Iterable[Parameter], params:dict) -> torch.optim.RMSprop:
    return torch.optim.RMSprop(
        model_params, 
        lr=params['lr'], 
        alpha=params['alpha'], 
        eps=params['eps'], 
        weight_decay=params['weight_decay'],
        momentum=params['momentum']
    )


optmizers = {
    'adam': get_adam_optimizer,
    'adamw': get_adamw_optimizer,
    'rmsprop': get_rmsprop_optimizer
}

def get_optimizer(model_params:Iterable[Parameter], params:dict) -> torch.optim.Optimizer:
    """
    Create an optimizer based on the provided parameters.
    """
    optimizer_type = params['name']
    if optimizer_type not in optmizers:
        raise ValueError(f"Optimizer type '{optimizer_type}' is not supported.")
    
    optimizer = optmizers[optimizer_type]
    
    return optimizer(model_params, params)


"""
Loss functions
"""
def get_cross_entropy_loss(params:dict, device) -> torch.nn.CrossEntropyLoss:
    weight = params['weight']
    if params['weight'] is not None:
        weight = torch.tensor(weight, dtype=torch.float).to(device)

    reduction = params['reduction'] if params['reduction'] is not None else 'mean'
    label_smoothing = params['label_smoothing'] if params['label_smoothing'] is not None else 0.0

    return torch.nn.CrossEntropyLoss(
        weight=weight,
        reduction=reduction,
        label_smoothing=label_smoothing
    )


def get_focal_loss(params:dict, device):
    gamma = params['gamma'] if params['gamma'] is not None else 2.0
    alpha = params['alpha'] if params['alpha'] is not None else None
    reduction = params['reduction'] if params['reduction'] is not None else 'mean'
    weight = params['weight']

    return FocalLoss(
        to_onehot_y=True,
        gamma=gamma,
        alpha=alpha,
        reduction=reduction,
        weight=weight
    )


loss_functions ={
    'cross_entropy_loss': get_cross_entropy_loss,
    'focal_loss': get_focal_loss,
}

def get_loss(params:dict, device) -> torch.nn.Module:
    """
    Create a loss function based on the provided parameters.
    """
    loss_name = params['name']
    if loss_name not in loss_functions:
        raise ValueError(f"Loss function type '{loss_name}' is not supported.")
    
    loss_function = loss_functions[loss_name]
    
    return loss_function(params, device)




"""
Networks
"""
resnets = [
    'resnet10',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet10p',
    'resnet18p',
    'resnet34p',
    'resnet50p',
    'resnet101p'
]
densenets = [
    'densenet121',
    'densenet169',
    'densenet201'
]
efficientnets = [
    'efficientnet-b0',
    'efficientnet-b1',
    'efficientnet-b2',
    'efficientnet-b3',
    'efficientnet-b4',
    'efficientnet-b5',
    'efficientnet-b6',
    'efficientnet-b7',
    'efficientnet-b8'
]
custom = [
    'AD3DCNN'
]

def get_network(params:dict):
    model_name = params['name']
    if model_name in resnets:
        return get_resnet_model(params)
    if model_name in densenets:
        return get_densenet_model(params)
    if model_name in efficientnets:
        return get_efficientnet_model(params)
    if model_name in custom:
        return get_custom_net(params)
    return None



"""
Scheduler
"""
def get_reduce_lr_on_plateau(params:dict, optimizer:torch.optim) -> torch.optim.lr_scheduler:
    params['factor']
    params['patience']
    params['min_lr']
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=params['factor'],
        patience=params['patience'],
        min_lr=params['min_lr']
    )
    

schedulers = {
   'ReduceLROnPlateau' : get_reduce_lr_on_plateau
}

def get_scheduler(params:dict, optimizer:torch.optim):
    scheduler_name = params['name']
    if scheduler_name not in schedulers:
        raise ValueError(f"Scheduler type '{scheduler_name}' is not supported.")
    
    scheduler = schedulers[scheduler_name]
    
    return scheduler(params, optimizer)
