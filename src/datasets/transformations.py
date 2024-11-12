# transformations.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 23.10.24

import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchio as tio
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms


class twoDTransforms:
    def __init__(self, input_size: Tuple[int, int], augmentations: Dict[str, Any]):
        self.input_size = input_size
        self.augmentations = augmentations

    def __call__(self, image: Image.Image, label: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:

        # Convert image and label to tensors
        image = F.to_tensor(image)
        label = F.to_tensor(label)

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

        # Resize both image and label
        image = F.resize(image, self.input_size)
        label = F.resize(label, self.input_size, interpolation=F.InterpolationMode.NEAREST)

        # Normalize the image (not the label)
        if 'mean' in self.augmentations and 'std' in self.augmentations:
            mean = self.augmentations['mean']
            std = self.augmentations['std']
            image = F.normalize(image, mean=mean, std=std)

        return image, label


class threeDTransforms:

    def __init__(self, input_size: Tuple[int, int, int], augmentations: Dict[str, Any]):
        self.input_size = input_size
        self.augmentations = augmentations

        transforms = []

        # Random flip
        if self.augmentations.get('random_flip', False):
            axes = self.augmentations.get('flip_axes', (0, 1, 2))
            transforms.append(tio.RandomFlip(axes=axes))

        # Random affine transformation
        if self.augmentations.get('random_affine', False):
            scales = self.augmentations.get('scales', (0.9, 1.1))
            degrees = self.augmentations.get('rotation_degrees', (-10, 10))
            transforms.append(tio.RandomAffine(scales=scales, degrees=degrees))

        # Random noise
        if self.augmentations.get('random_noise', False):
            std = self.augmentations.get('noise_std', (0, 0.1))
            transforms.append(tio.RandomNoise(std=std))

        # Resize
        transforms.append(tio.Resize(self.input_size))

        # Compose transforms
        self.transform = tio.Compose(transforms)

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        # Ensure image and label tensors are 4D: (C, D, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if label.dim() == 3:
            label = label.unsqueeze(0)

        # Scale image to [0, 1] if it's an 8-bit image
        if image.dtype == torch.uint8:
            image = image.float().div(255)
        else:
            image = image.float()

        if label.dtype == torch.uint8:
            label = label.float().div(255)
        else:
            label = label.float()

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image),
            label=tio.LabelMap(tensor=label),
        )

        transformed = self.transform(subject)
        image = transformed['image'].data
        label = transformed['label'].data

        # Normalize the image (not the label)
        if 'mean' in self.augmentations and 'std' in self.augmentations:
            mean = self.augmentations['mean']
            std = self.augmentations['std']
            image = (image - mean) / std

        return image, label


