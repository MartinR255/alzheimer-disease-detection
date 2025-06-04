import torch
from torch.utils.data import IterableDataset
import os
import random

class BatchDataset(IterableDataset):
    def __init__(self, folder_path:str):
        self._chunk_file_paths = [] 
        for file_name in os.listdir(folder_path):
            self._chunk_file_paths.append(os.sep.join([folder_path, file_name]))


    def parse_batch(self, file_path):
        batch_tensor = torch.load(file_path)  # Shape: (100, D, H, W) or (100, 1, D, H, W)
        for i in range(batch_tensor.size(0)):
            sample = batch_tensor[i]
            if sample.dim() == 3:
                sample = sample.unsqueeze(0)  # Add channel dimension
            yield sample


    def __iter__(self):
        for file_path in self.batch_file_paths:
            yield from self.parse_batch(file_path)


class BatchDataset(IterableDataset):

    def __init__(self, folder_path:str, batch_size, shuffle=True):
        self._batch_size = batch_size
        self._shuffle = shuffle

        self._chunk_file_paths = [] 
        for file_name in os.listdir(folder_path):
            self._chunk_file_paths.append(os.sep.join([folder_path, file_name]))
       

    def __iter__(self):
        chunk_paths = self._chunk_file_paths.copy()
        if self.shuffle:
            random.shuffle(chunk_paths)

        for chunk_path in chunk_paths:
            data = torch.load(chunk_path)
            images, labels = data['images'], data['labels']
            num_samples = images.shape[0]

            indices = list(range(num_samples))
            if self.shuffle:
                random.shuffle(indices)

            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                yield images[batch_indices], labels[batch_indices]