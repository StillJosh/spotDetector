# deepSpotDataset.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 17.10.24


# utils/dataset.py

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import polars as pl
import torch
from torch.utils.data import Dataset

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
        self.data_dir = data_dir
        self.dataset_hdf5 = None
        self.patches = None
        self.labels = None

        self.input_size = input_size
        self.metadata = metadata
        self.debug = debug

        if augmentations is None:
            augmentations = {}

        self.transform = threeDTransforms(self.input_size, augmentations)


    def __len__(self) -> int:
        if self.debug:
            return min(2, len(self.metadata))
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.dataset_hdf5 is None:
            self.dataset_hdf5 = h5py.File(self.data_dir, 'r')
            self.patches = self.dataset_hdf5['patches']
            self.labels = self.dataset_hdf5['labels']

        num_image = self.metadata['num_image'][idx]
        image, label = self.patches[num_image], self.labels[num_image]
        image, label = self.transform(image, label)

        return image, label, idx

