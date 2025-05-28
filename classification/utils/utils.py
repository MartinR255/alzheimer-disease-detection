import os
import yaml
import json
import numpy as np
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torchvision.transforms import Normalize

from torch.nn import Parameter
from memory_dataset import MemoryDataset
from monai.data import ImageDataset
from monai.losses import FocalLoss

from typing import Iterable

from monai.transforms import (
    Compose, 
    LoadImage, 
    Resize, 
    ScaleIntensity, 
    EnsureChannelFirst, 
    Spacing, 
    CropForeground,
    ScaleIntensity
)

from sklearn.model_selection import train_test_split

__all__ = [
    'copy_config',
    'load_data',
    'get_transform',
    'get_memory_dataset',   
    'get_image_dataset',
    'save_dataset_to_file',
    'stratified_split',
    'load_yaml_config',
    'get_optimizer',
    'get_loss'
]



def copy_config(file_path:str, new_file_path:str):
    folder_path = os.path.dirname(new_file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

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
        return x > 0
     
    data_transform = Compose(
        [
            LoadImage(reader="monai.data.ITKReader"),
            EnsureChannelFirst(),
            Spacing(pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
            ScaleIntensity(minv=0, maxv=1.0, dtype=torch.float16),
            CropForeground(
                select_fn=select_fn,
                allow_smaller=False,
                margin=0,
            ),
            Resize(spatial_size=(128, 128, 128)),
            Normalize(mean=84.28270578018392, std=250.33769250046794),
            ScaleIntensity(minv=0, maxv=1.0, dtype=torch.float16)
        ]
    )
    return data_transform


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
    if params['weight'] is not None:
        weight = torch.tensor(params['weights'], dtype=torch.float).to(device)

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

    return FocalLoss(
        to_onehot_y=True,
        gamma=gamma,
        alpha=alpha,
        reduction=reduction
    )


loss_functions ={
    'cross_entropy_loss': get_cross_entropy_loss,
    'focal_loss': get_cross_entropy_loss,
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

