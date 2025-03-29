import os
import json
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from memory_dataset import MemoryDataset
from monai.data import ImageDataset

from monai.transforms import (
    Compose, 
    LoadImage, 
    Resize, 
    ScaleIntensity, 
    EnsureChannelFirst, 
    Orientation, 
    Spacing, 
    RandFlip, 
    RandGaussianNoise, 
    RandAdjustContrast, 
    CropForeground,
    SpatialPad, 
    ScaleIntensity, 
    RandShiftIntensity,
    ToTensor
)


from sklearn.model_selection import train_test_split


def load_data(image_dataset_path: str, dataset_partiton_path: str):
    """
    Load dataset from a JSON file.
    
    Args:
        image_dataset_path (str): Path to the directory containing image files.
        dataset_partiton_path (str): Path to the JSON file containing image filenames and labels.
    
    Returns:
        list: List of dictionaries, each containing 'image' path and 'label' for a sample.
    """
    images = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int64)
    with open(dataset_partiton_path, 'r') as data_file:
        data = json.load(data_file)
        for image, label in data.items():
            images = np.append(images, os.sep.join([image_dataset_path, image]))
            labels = np.append(labels, label)
    return images, labels



def get_transform():
    """
    Creates a transformation pipeline for image preprocessing before feeding to the model.

    Returns:
        Compose: A transformation pipeline that includes loading, scaling, and resizing images.
    """
    def select_fn(x):
        return x > -1
     
    data_transform = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensity(minv=-1, maxv=1.0, dtype=torch.float16), # dtype=torch.float16 
            Spacing(pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
            
            CropForeground(
                select_fn=select_fn,
                allow_smaller=False,
                margin=0,
            ),
            # SpatialPad(
            #     spatial_size=(184, 184, 184),  
            #     value=-1.0
            # ),
            Resize(spatial_size=(128, 128, 128)),
            ToTensor(dtype=torch.float16)
        ]
    )

    return data_transform


def get_memory_dataset(dataset_path:str = None):
    """
    Creates a MemoryDataset from file.
    
    Args:
        dataset_path (str): Path to the dataset file where tensors of images and labels are stored.

    Returns:
        MemoryDataset: Dataset loaded in memory.
    """
    return MemoryDataset(path=dataset_path)


def get_image_dataset(image_dataset_path:str, dataset_partiton_path:str):
    """
    
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


def stratified_split(images, labels, ratios:tuple = (0.8, 0.1, 0.1), seed:int = 42):
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



