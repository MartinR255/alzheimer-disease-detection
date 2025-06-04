from monai.transforms import Transform
import numpy as np
import torch

class IntensityNormalization(Transform):
    """
    Normalize the intensity of a volume:
    - Clips values to the `clip_ratio` percentile.
    - Scales to [-1, 1].
    """
    def __init__(self, clip_ratio: float = 99.5):
        if not (0.0 < clip_ratio <= 100.0):
            raise ValueError("clip_ratio must be in (0.0, 100.0].")
        self.clip_ratio = clip_ratio


    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if not isinstance(volume, (np.ndarray, torch.Tensor)):
            raise TypeError("Input must be a numpy ndarray or torch tensor.")

        volume = volume - np.min(volume)  # Shift min to zero
        intensity_clip_threshold = np.percentile(volume, self.clip_ratio)

        if intensity_clip_threshold == 0:
            raise ValueError("Percentile resulted in zero, cannot divide by zero.")

        normalized = np.clip(volume / intensity_clip_threshold, 0, 1) * 2 - 1
        return normalized
    
