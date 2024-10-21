# deepSpotDataset.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 17.10.24

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import boto3
import polars as pl
from PIL import Image

from .deepSpotDataset import DeepSpotDataset


class DeepSpotDatasetS3(DeepSpotDataset):
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
            s3_bucket: str = 'tunamlbucket',
    ):
        super().__init__(data_dir, metadata, input_size, augmentations, debug)
        self.s3 = None

    def load_image(self, row: pl.Series) -> Image:
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
        if self.s3 is None:
            self.s3 = boto3.client('s3')

        #file_path = row[0, 'file_path']
        file_path = 'spotDetection/' + row[0, 'file_path']
        # Load image from S3
        obj = self.s3.get_object(Bucket='tunamlbucket', Key=file_path)
        img_data = obj['Body'].read()
        return Image.open(BytesIO(img_data))

