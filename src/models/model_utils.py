# model_utils.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 15.11.24
import numpy as np
import torch


def split_array(array: np.ndarray | torch.Tensor, shape: tuple) -> np.ndarray | torch.Tensor:
    """
    Splits an array into chunks of the specified shape, and stacks them.

    Parameters
    ----------
        array: np.ndarray,
            The array to split.
        shape: tuple,
            The shape of the chunks.

    Returns
    -------
        np.ndarray | torch.Tensor,
            The array of chunks.
    """

    # Check if the shape is compatible
    if any(array.shape[i] % shape[i] != 0 for i in range(len(shape))):
        raise ValueError("Array dimensions must be divisible by the desired shape.")

    # Calculate the number of splits along each dimension
    num_splits = tuple(array.shape[i] // shape[i] for i in range(len(shape)))

    # Generate sub-arrays using slicing
    if isinstance(array, torch.Tensor):
        results = torch.zeros((np.prod(num_splits), *shape), dtype=array.dtype)
    else:
        results = np.zeros((np.prod(num_splits), *shape), dtype=array.dtype)
    for i, idx in enumerate(np.ndindex(num_splits)):
        slices = tuple(slice(idx[i] * shape[i], (idx[i] + 1) * shape[i]) for i in range(len(shape)))
        results[i] = array[slices]

    return results


import numpy as np
import torch


def merge_chunks(chunks: np.ndarray | torch.Tensor, original_shape: tuple) -> np.ndarray | torch.Tensor:
    """
    Merges an array of chunks back into the original array.

    Parameters
    ----------
        chunks: np.ndarray | torch.Tensor,
            The array of chunks.
        original_shape: tuple,
            The shape of the original array.

    Returns
    -------
        np.ndarray | torch.Tensor,
            The reconstructed array.
    """
    # Infer the shape of individual chunks and the number of splits
    chunk_shape = chunks.shape[1:]  # Shape of each chunk
    num_splits = tuple(o // c for o, c in zip(original_shape, chunk_shape))

    # Reshape the chunks array into a grid of sub-arrays
    grid_shape = (*num_splits, *chunk_shape)
    if isinstance(chunks, torch.Tensor):
        reshaped_chunks = chunks.view(*grid_shape)
        # Merge chunks along each dimension
        merged_array = reshaped_chunks.permute(0, 2, 1, 3).contiguous().view(original_shape)
    else:
        reshaped_chunks = chunks.reshape(*grid_shape)
        # Merge chunks along each dimension
        axes_order = [i for pair in zip(range(len(num_splits)), range(len(num_splits), len(grid_shape))) for i in pair]
        merged_array = reshaped_chunks.transpose(*axes_order).reshape(original_shape)

    return merged_array


