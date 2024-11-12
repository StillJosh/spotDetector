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
import polars as pl

from datasets.transformations import twoDTransforms, threeDTransforms


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
        metadata: pl.DataFrame,
        input_size: Tuple[int, int] = (256, 256),
        augmentations: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.metadata = metadata
        self.debug = debug

        if augmentations is None:
            augmentations = {}

        self.transform = threeDTransforms(self.input_size, augmentations)

    def load_image(self, row: pl.DataFrame) -> Tuple[Image, Image]:
        """
        Load an image from the specified path and convert it to grayscale.

        Parameters
        ----------
        image_path: str
            Path to the image file.

        Returns
        -------
        Image
            Grayscale image.
        """
        file_path = self.data_dir / row[0, 'file_path']
        label_path = self.data_dir / row[0, 'label_path']
        return Image.open(file_path), Image.open(label_path)


    def __len__(self) -> int:
        if self.debug:
            return min(2, len(self.metadata))
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata[idx]
        image, label = self.load_image(row)
        image, label = self.transform(image, label)

        return image, image.clone()


