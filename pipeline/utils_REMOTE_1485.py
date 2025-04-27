import os

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


def load_model(model, model_path:str) -> None:
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model



def get_transform() -> Compose:
    """
    Creates a transformation pipeline for image preprocessing before feeding to the model.
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
            ToTensor(dtype=torch.float32)
        ]
    )

    return data_transform