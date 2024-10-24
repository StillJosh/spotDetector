# transformations.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 23.10.24

import random
from typing import Any, Dict, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms


class CustomTransforms:
    def __init__(self, input_size: Tuple[int, int], augmentations: Dict[str, Any]):
        self.input_size = input_size
        self.augmentations = augmentations

    def __call__(self, image: Image.Image, label: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize both image and label
        image = F.resize(image, self.input_size)
        label = F.resize(label, self.input_size, interpolation=F.InterpolationMode.NEAREST)

        # Random horizontal flip
        if self.augmentations.get('horizontal_flip', False):
            if random.random() > 0.5:
                image = F.hflip(image)
                label = F.hflip(label)

        # Random vertical flip
        if self.augmentations.get('vertical_flip', False):
            if random.random() > 0.5:
                image = F.vflip(image)
                label = F.vflip(label)

        # Random rotation
        if self.augmentations.get('rotation', False):
            degrees = self.augmentations.get('rotation_degrees', 90)
            angle = random.uniform(-degrees, degrees)
            image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
            label = F.rotate(label, angle, interpolation=F.InterpolationMode.NEAREST)

        # Apply color jitter to the image only
        if self.augmentations.get('color_jitter', False):
            brightness = self.augmentations.get('brightness', 0.2)
            contrast = self.augmentations.get('contrast', 0.2)
            saturation = self.augmentations.get('saturation', 0.2)
            hue = self.augmentations.get('hue', 0.1)
            self.color_jitter = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
            image = self.color_jitter(image)

        # Convert image and label to tensors
        image = F.to_tensor(image)
        label = F.pil_to_tensor(label).squeeze().long()

        # Normalize the image (not the label)
        if 'mean' in self.augmentations and 'std' in self.augmentations:
            mean = self.augmentations['mean']
            std = self.augmentations['std']
            image = F.normalize(image, mean=mean, std=std)

        return image, label
