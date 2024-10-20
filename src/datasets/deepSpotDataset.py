# deepSpotDataset.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 17.10.24


# utils/dataset.py

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DeepSpotDataset(Dataset):
    """
    Custom dataset for loading 2D images with configurable augmentations.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory containing image files.
    input_size : Tuple[int, int]
        Desired input size (height, width).
    augmentations : Dict[str, Any]
        Dictionary specifying augmentations and their parameters.
    debug : bool
        Whether to run the dataset in debug mode. If True, only two batches will be loaded.

    Attributes
    ----------
    images : List[Path]
        List of image file paths.
    labels : Optional[List[Path]]
        List of label file paths.
    transform : Callable
        Transformation pipeline for the images.
    label_transform : Callable
        Transformation pipeline for the labels.
    debug : bool
        Whether the dataset is in debug mode. If True, only two samples will be loaded.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        input_size: Tuple[int, int] = (256, 256),
        augmentations: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.images = sorted(self.data_dir.glob('*.*'))
        self.debug = debug

        if augmentations is None:
            augmentations = {}

        self.transform = self.get_transform(augmentations, is_label=False)

    def __len__(self) -> int:
        if self.debug:
            return min(2, len(self.images))
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')
        image = self.transform(image)

        return image, image.clone()

    def get_transform(self, augmentations: Dict[str, Any], is_label: bool) -> Callable:
        """
        Create a transformation pipeline based on the augmentations specified.

        Parameters
        ----------
        augmentations : Dict[str, Any]
            Dictionary specifying augmentations and their parameters.
        is_label : bool
            Whether the transformation is for labels.

        Returns
        -------
        Callable
            Transformation pipeline.
        """
        transform_list = []

        # Resize
        resize = transforms.Resize(self.input_size)
        transform_list.append(resize)

        # Data augmentations (only applied to images, not labels)
        if not is_label:
            if augmentations.get('horizontal_flip', False):
                transform_list.append(transforms.RandomHorizontalFlip())

            if augmentations.get('vertical_flip', False):
                transform_list.append(transforms.RandomVerticalFlip())

            if augmentations.get('rotation', False):
                degrees = augmentations.get('rotation_degrees', 90)
                transform_list.append(transforms.RandomRotation(degrees))

            if augmentations.get('color_jitter', False):
                brightness = augmentations.get('brightness', 0.2)
                contrast = augmentations.get('contrast', 0.2)
                saturation = augmentations.get('saturation', 0.2)
                hue = augmentations.get('hue', 0.1)
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue,
                    )
                )

        # To Tensor
        if not is_label:
            transform_list.append(transforms.ToTensor())
            # Normalize if mean and std are provided
            if 'mean' in augmentations and 'std' in augmentations:
                mean = augmentations['mean']
                std = augmentations['std']
                transform_list.append(transforms.Normalize(mean=mean, std=std))
        else:
            transform_list.append(transforms.PILToTensor())
            # For labels, ensure the tensor is of type long (for loss functions like CrossEntropyLoss)
            transform_list.append(transforms.Lambda(lambda x: x.squeeze().long()))

        return transforms.Compose(transform_list)
