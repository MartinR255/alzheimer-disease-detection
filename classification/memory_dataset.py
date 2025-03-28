import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np


 
class MemoryDataset(Dataset):
    
    def __init__(self, images, labels, transform, path=None):
        """
        Args:
            images (list): List of image paths
            labels (list): List of labels
            transform (callable): transform to be applied
            path (str): path to the file with the data
        """
        if len(images) != len(labels):
            raise ValueError(f"Number of images ({len(images)}) does not match number of labels ({len(labels)})")
        
        self._labels = labels
        if path is not None and os.path.exists(path):
            self._data = torch.load(path)
        else:
            self._data = torch.empty([len(images), 1, 128, 128, 128], dtype=torch.float16) 
            for i, img_path in enumerate(images):
                img_tensor =  transform(img_path)
                self._data[i] = img_tensor.to(dtype=torch.float16)
            
            if path is not None:
                torch.save(self._data, path)

        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self._labels)
    

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Args:
            idx (int): Index of the sample to get
        Returns:
            tuple: A tuple containing a tensor of shape [1, 128, 128, 128] and a label
        """
        img_tensor = self._data[idx].to(dtype=torch.float32)
        label = self._labels[idx] 
        return tuple([img_tensor, label])