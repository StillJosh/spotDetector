# inference.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 16.10.24

from pathlib import Path
from typing import List, Optional, Tuple, Union, _SpecialForm

import numpy as np
import tifffile
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from src import plotting as pl


class Inferencer:
    """
    A class to handle inference for image models.

    Parameters
    ----------
    model : nn.Module
        The trained PyTorch model.
    device : torch.device
        The device to run inference on.

    Attributes
    ----------
    model : nn.Module
        The model set to evaluation mode.
    device : torch.device
        The device to run inference on.
    """

    def __init__(self, model: nn.Module, images: torch.Tensor, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.images = images
        self.device = device

    def infer_on_patches(
            self,
            images: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Run inference on random preprocessed patches of images.

        Parameters
        ----------
        images : torch.Tensor
            The input images tensor of shape (B, C, H, W).
        Returns
        -------
        List[torch.Tensor]
            List of output image patches.
        """
        images = self.images.clone() if images is None else images
        with torch.no_grad():
            images = images.to(self.device)
            output = self.model(images)

        return output.cpu()

    def infer_on_full_image(
            self,
            image_path: Union[str, Path],
            patch_size: Tuple[int, int],
            channels: Optional[List[int]] = None,
            overlap: int = 0,
    ) -> np.ndarray:
        """
        Run inference on a full image by splitting it into patches,
        running inference on each patch, and stitching the results back together.

        Parameters
        ----------
        image_path : Union[str, Path]
            Path to the input image file.
        patch_size : Tuple[int, int]
            Size of each patch (height, width).
        channels : Optional[List[int]], default=None
            List of channels to select from the image.
            If None, all channels are used.
        overlap : int, default=0
            Number of pixels to overlap between patches.

        Returns
        -------
        np.ndarray
            The stitched output image with the same metadata as the input.
        """
        # Load image with tifffile to preserve metadata
        image = tifffile.imread(image_path)
        tiff = tifffile.TiffFile(image_path)
        metadata = tiff.pages[0].tags

        # Select relevant channels
        if channels is not None:
            image = image[channels, ...]
        else:
            # Assume first dimension is channels
            pass

        # Check image dimensions
        image_shape = image.shape
        if len(image_shape) == 4:
            # Image has dimensions (C, Z, Y, X)
            c, z_max, y_max, x_max = image_shape
            output_image = np.zeros_like(image)
            for z in range(z_max):
                image_slice = image[:, z, :, :]
                output_slice = self._infer_on_slice(
                    image_slice, patch_size, overlap
                )
                output_image[:, z, :, :] = output_slice
        elif len(image_shape) == 3:
            # Image has dimensions (C, Y, X)
            output_image = self._infer_on_slice(
                image, patch_size, overlap
            )
        else:
            raise ValueError("Unsupported image dimensions.")

        # Optionally, save or handle metadata here
        # For example, you can save the output image with metadata using tifffile
        # tifffile.imwrite('output_image.tif', output_image, metadata=metadata)

        return output_image

    def _infer_on_slice(
            self,
            image_slice: np.ndarray,
            patch_size: Tuple[int, int],
            overlap: int = 0,
    ) -> np.ndarray:
        """
        Run inference on a single slice (C, Y, X).

        Parameters
        ----------
        image_slice : np.ndarray
            The image slice of shape (C, Y, X).
        patch_size : Tuple[int, int]
            Size of each patch (height, width).
        overlap : int, default=0
            Number of pixels to overlap between patches.

        Returns
        -------
        np.ndarray
            The output image slice after inference.
        """
        # Split image slice into patches
        patches, coords = self._split_image_into_patches(
            image_slice, patch_size, overlap
        )

        # Run inference on patches
        outputs = []
        with torch.no_grad():
            for patch in patches:
                patch_tensor = self._preprocess_patch(patch)
                patch_tensor = patch_tensor.to(self.device)
                output = self.model(patch_tensor)
                outputs.append(output.cpu().numpy())

        # Stitch patches back together
        output_slice = self._stitch_patches_back(
            outputs, coords, image_slice.shape, overlap
        )

        return output_slice

    def _split_image_into_patches(
            self,
            image: np.ndarray,
            patch_size: Tuple[int, int],
            overlap: int = 0,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Split the image into patches.

        Parameters
        ----------
        image : np.ndarray
            The input image array of shape (C, Y, X).
        patch_size : Tuple[int, int]
            Size of each patch (height, width).
        overlap : int, default=0
            Number of pixels to overlap between patches.

        Returns
        -------
        patches : List[np.ndarray]
            List of image patches.
        coords : List[Tuple[int, int]]
            List of coordinates (y, x) for each patch.
        """
        c, y_max, x_max = image.shape
        patch_height, patch_width = patch_size
        patches = []
        coords = []

        y_steps = self._get_patch_indices(y_max, patch_height, overlap)
        x_steps = self._get_patch_indices(x_max, patch_width, overlap)

        for y in y_steps:
            for x in x_steps:
                patch = image[:, y:y + patch_height, x:x + patch_width]
                patches.append(patch)
                coords.append((y, x))

        return patches, coords

    def _get_patch_indices(
            self,
            max_size: int,
            patch_size: int,
            overlap: int,
    ) -> List[int]:
        """
        Compute the starting indices for patches along one dimension.

        Parameters
        ----------
        max_size : int
            The size of the dimension (height or width).
        patch_size : int
            Size of each patch along this dimension.
        overlap : int
            Number of pixels to overlap between patches.

        Returns
        -------
        List[int]
            List of starting indices.
        """
        indices = []
        step = patch_size - overlap
        i = 0
        while i + patch_size <= max_size:
            indices.append(i)
            i += step
        if i < max_size:
            indices.append(max_size - patch_size)
        return indices

    def _preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single patch for inference.

        Parameters
        ----------
        patch : np.ndarray
            The image patch of shape (C, H, W).

        Returns
        -------
        torch.Tensor
            The preprocessed patch tensor.
        """
        # Convert to torch tensor
        patch_tensor = torch.from_numpy(patch).float()  # shape (C, H, W)

        # Apply any necessary transformations (e.g., normalization)
        # If you have mean and std for normalization, apply here
        # For example:
        # normalize = transforms.Normalize(mean=[...], std=[...])
        # patch_tensor = normalize(patch_tensor)

        return patch_tensor.unsqueeze(0)  # Add batch dimension

    def _stitch_patches_back(
            self,
            outputs: List[np.ndarray],
            coords: List[Tuple[int, int]],
            image_shape: Tuple[int, int, int],
            overlap: int = 0,
    ) -> np.ndarray:
        """
        Stitch the output patches back into a full image.

        Parameters
        ----------
        outputs : List[np.ndarray]
            List of output arrays from inference, each of shape (C_out, H, W).
        coords : List[Tuple[int, int]]
            List of coordinates (y, x) for each patch.
        image_shape : Tuple[int, int, int]
            The shape of the original image (C_in, Y, X).
        overlap : int, default=0
            Number of pixels that were overlapped between patches.

        Returns
        -------
        np.ndarray
            The stitched output image of shape (C_out, Y, X).
        """
        c_in, y_max, x_max = image_shape
        # Get output channels from first output
        output_sample = outputs[0]
        if output_sample.ndim == 4:
            output_sample = output_sample[0]
        c_out = output_sample.shape[0]
        output_image = np.zeros((c_out, y_max, x_max), dtype=np.float32)
        weight_image = np.zeros((1, y_max, x_max), dtype=np.float32)

        for output, (y, x) in zip(outputs, coords):
            # Remove batch dimension if present
            if output.ndim == 4:
                output = output[0]
            # output shape: (C_out, H, W)
            _, h, w = output.shape
            output_image[:, y:y + h, x:x + w] += output
            weight_image[:, y:y + h, x:x + w] += 1

        # Avoid division by zero
        weight_image[weight_image == 0] = 1
        output_image /= weight_image

        return output_image

    def plot_inference_comparison(
            self,
            images: Optional[torch.Tensor] = None,
            save_path: Union[Path, str] = None,
            show: bool = False
    ) -> plt.Figure:
        """
        Plot the original image and its corresponding inference image side by side.

        Parameters
        ----------
        images : Optional[torch.Tensor], default=None
            The input images tensor of shape (B, C, H, W).
            If None, uses `self.images`.
        save_path : Union[Path, str], default=None
            If provided, saves the figure to the specified file path.
        show : bool, default=False
            If True, displays the plot.

        Returns
        -------
        Optional[plt.Figure]
            Returns the figure object if neither saving nor showing, otherwise returns None.
        """
        images = self.images.clone() if images is None else images
        output = self.infer_on_patches(images)

        # Squeeze channels if they are singleton
        if images.shape[1] == 1:
            images = images.squeeze(1)
            output = output.squeeze(1)

        # Convert the tensors to NumPy arrays for plotting
        images = images.cpu().numpy()
        output = output.cpu().numpy()

        # Use plot_multiple_comparisons to create the plot
        fig = pl.plot_multiple_comparisons(
            original_images=images,
            inference_images=output,
            save_path=save_path,
            show=show
        )

        # Return the figure object if it was neither shown nor saved
        return fig

