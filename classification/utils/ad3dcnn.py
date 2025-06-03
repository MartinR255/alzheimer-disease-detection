import torch
import torch.nn as nn
import torch.nn.functional as F

class AD3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(2),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


    
def get_custom_net(params:dict) -> torch.nn.Module:
    """
    Create x model with given parameters.
    
    Args:
        params (dict): Dictionary containing model parameters.
        
    Returns:
        torch.nn.Module: x model instance.
    """
    return AD3DCNN(
        in_channels=params['n_input_channels'], 
        num_classes=params['num_classes']
    )





