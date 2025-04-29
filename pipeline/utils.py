import os
import yaml
from pathlib import Path
from pydicom.misc import is_dicom



os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
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


def load_yaml_config(file_path:str) -> dict:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def find_dicom_directories(root_path) -> list:
    dicom_dirs = set()
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_path = Path(os.sep.join([dirpath, filename])).as_posix()
            try:
                if is_dicom(file_path):
                    dicom_dirs.add(Path(dirpath).as_posix())
                    break  
            except Exception:
                continue 
    return dicom_dirs
    


def del_file(file_path:str, logger):
    try:
        os.remove(file_path)
        logger.debug(f"File {file_path} deleted successfully.")
    except OSError as e:
        logger.error(f"Error deleting file {file_path}: {e.strerror}")


def load_model(model, model_path:str) -> None:
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def get_transform_clean_tensor() -> Compose:
    """
    Creates a transformation pipeline for image preprocessing before feeding to the model.
    """
    def select_fn(x):
        return x > -1
    
    data_transform = Compose(
        [
            LoadImage(reader="monai.data.ITKReader"),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensity(minv=-1, maxv=1.0, dtype=torch.float16), 
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
        ]
    )

    return data_transform


def get_transform_resample_tensor():
    data_transform = Compose(
        [   
            Resize(spatial_size=(128, 128, 128)),
            ScaleIntensity(minv=-1, maxv=1.0, dtype=torch.float32)
        ]
    )

    return data_transform
