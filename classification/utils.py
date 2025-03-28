import os
import json
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from memory_dataset import MemoryDataset

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

from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAUROC
)

from numpy import array

def load_data(dataset_path: str, dataset_partiton_path: str):
    """
    Load dataset from a JSON file.
    
    Args:
        dataset_path (str): Path to the directory containing image files.
        dataset_partiton_path (str): Path to the JSON file containing image filenames and labels.
    
    Returns:
        list: List of dictionaries, each containing 'image' path and 'label' for a sample.
    """
    # dataset = []
    images = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int64)
    with open(dataset_partiton_path, 'r') as data_file:
        data = json.load(data_file)
        # dataset = [{'image': os.sep.join([dataset_path, image]), 'label': label} for image, label in data.items()]
        for image, label in data.items():
            images = np.append(images, os.sep.join([dataset_path, image]))
            labels = np.append(labels, label)
    return images, labels



def select_fn(x):
    return x > -1


def get_dataset(images: np.array, labels: np.array, path: str = None):
    """
    Create a Dataset with appropriate transforms.
    
    Args:
        images (np.array): List of image paths.
        labels (np.array): List of labels.

    Returns:
        Dataset: Dataset with appropriate transforms applied.
    """
    base_transforms = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensity(minv=0, maxv=1.0, dtype=torch.float16), # dtype=torch.float16 
            Spacing(pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
            
            CropForeground(
                select_fn=select_fn,
                allow_smaller=False,
                margin=0,
            ),
            SpatialPad(
                spatial_size=(184, 184, 184),  
                value=-1.0
            ),
            Resize(spatial_size=(128, 128, 128)),
            ToTensor(dtype=torch.float16)
        ]
    )
      
    labels = torch.tensor(labels, dtype=torch.int64)
    
    ds = MemoryDataset(
        images=images, 
        labels=labels, 
        transform=base_transforms, 
        path=path
    )
    return ds



def create_metrics(num_classes: int = 5):
    """
    Create evaluation metrics for multiclass classification.
    
    Args:
        None
        
    Returns:
        dict: Dictionary containing initialized metrics:
            - 'accuracy': MulticlassAccuracy
            - 'precision': MulticlassPrecision
            - 'recall': MulticlassRecall
            - 'f1_score': MulticlassF1Score
            - 'auroc': MulticlassAUROC
    """
    accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
    precision = MulticlassPrecision(num_classes=num_classes, average='macro')
    recall = MulticlassRecall(num_classes=num_classes, average='macro')
    f1_score = MulticlassF1Score(num_classes=num_classes, average='macro')
    auroc = MulticlassAUROC(num_classes=num_classes, average='macro')

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score, 'auroc': auroc}


def reset_metrics(metrics):
    """
    Reset all metrics to their initial state.
    
    Args:
        metrics (dict): Dictionary of metric objects to reset.
        
    Returns:
        None
    """
    for metric in metrics.values():
        metric.reset()



def stratified_split(images, labels, ratios: tuple = (0.8, 0.1, 0.1), seed: int = 42):
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



