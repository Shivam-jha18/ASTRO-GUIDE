"""
Microgravity 6-DOF Data Augmentation Pipeline.
Applies physical zero-gravity transforms to raw image frames:
- 360-degree rotation invariance
- Floating drift and parallax distortion
- Cabin illumination shifts (ISS/BAS solar transit cycles)
- Sensor noise and motion blur
"""

import cv2
import random
import numpy as np
from typing import Tuple


class MicrogravityAugmentor:
    """
    Applies spatial and photometric augmentations to simulate spaceflight conditions.
    """

    def __init__(self, rotation_range: Tuple[float, float] = (-180.0, 180.0)):
        self.rotation_range = rotation_range

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Applies full microgravity augmentation suite."""
        aug = image.copy()

        # 1. 6-DOF Microgravity Rotation (Arbitrary roll angle)
        if random.random() > 0.3:
            angle = random.uniform(*self.rotation_range)
            h, w = aug.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 2. Lighting shift / Cabin solar reflection
        if random.random() > 0.4:
            alpha = random.uniform(0.75, 1.25)  # Contrast
            beta = random.randint(-20, 25)       # Brightness
            aug = np.clip(alpha * aug.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # 3. Microgravity Floating Motion Blur
        if random.random() > 0.6:
            k_size = random.choice([3, 5])
            aug = cv2.GaussianBlur(aug, (k_size, k_size), 0)

        # 4. Sensor grain
        if random.random() > 0.5:
            noise = np.random.normal(0, random.uniform(2, 6), aug.shape).astype(np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return aug
