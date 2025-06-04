import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torch.utils.data import Dataset


class MemoryDataset(Dataset):
    
    def __init__(self, path:str):
        """
        Args:
            path (str): Path to the data file
        """
        data = torch.load(path)
        self._images, self._labels = data['images'], data['labels']
       
        
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
        img_tensor = self._images[idx].to(dtype=torch.float32)
        label = self._labels[idx] 
        return tuple([img_tensor, label])
    
