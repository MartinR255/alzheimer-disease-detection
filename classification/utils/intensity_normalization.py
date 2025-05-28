from monai.transforms import Transform
import numpy as np

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
        if not isinstance(volume, np.ndarray):
            raise TypeError("Input must be a numpy ndarray.")

        volume = volume - np.min(volume)  # Shift min to zero
        volume_max = np.percentile(volume, self.clip_ratio)

        if volume_max == 0:
            raise ValueError("Percentile resulted in zero, cannot divide by zero.")

        normalized = np.clip(volume / volume_max, 0, 1) * 2 - 1
        return normalized
    
