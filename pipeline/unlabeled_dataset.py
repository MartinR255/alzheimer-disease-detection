from torch.utils.data import Dataset

from monai.transforms import Compose 

 
class UnlabeledDataset(Dataset):
    
    def __init__(self, image_paths:list, transform:Compose):
        """
        Args:
            path (str): Path to the data file
        """
        self._image_paths = image_paths
        self._tranform = transform
       
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self._image_paths)
    

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Args:
            idx (int): Index of the sample to get
        Returns:
            tuple: A tuple containing a tensor of shape [1, 128, 128, 128] and a image path
        """
        img_path = self._image_paths[idx]
        img_tensor = self._tranform(img_path)
        return tuple([img_tensor, img_path])
    
