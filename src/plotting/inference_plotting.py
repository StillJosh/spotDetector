# inference_plotting.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 18.10.24


from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plotting.plotting_utils import get_auto_contrast

color_dict = {
    'red': LinearSegmentedColormap.from_list('RedLUT', [(0, 0, 0), (1, 0, 0)]),
    'green': LinearSegmentedColormap.from_list('GreenLUT', [(0, 0, 0), (0, 1, 0)]),
    'cyan': LinearSegmentedColormap.from_list('CyanLUT', [(0, 0, 0), (0, 1, 1)])
}

def plot_inference_comparison(
        original_image: np.ndarray,
        labeled_image: np.ndarray,
        inference_image: np.ndarray,
        cmap: Optional[str] = 'viridis',
        title: Optional[str] = None,
        axes: Optional[Union[plt.Axes, np.ndarray]] = None,
        save_path: Optional[str] = None
) -> None:
    """
    Plot the original image and its corresponding inference image side by side.

    Parameters
    ----------
    original_image : np.ndarray
        The original image with shape (Y, X).
    labeled_image : np.ndarray
        The labeled image with shape (Y, X).
    inference_image : np.ndarray
        The inference image with shape (Y, X).
    cmap : Optional[str], default=viridis
        colormap to use for the original_image. If None, viridis will be used.
    title : Optional[str], default=None
        Title for the plot.
    axes : Optional[Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]], default=None
        If provided, the images will be plotted on these axes.
        Otherwise, new axes will be created.
    save_path : Optional[str], default=None
        If provided, saves the figure to the specified file path.
    """
    if original_image.shape != inference_image.shape:
        raise ValueError("Original image and inference image must have the same shape.")

    # Create figure and axes if not provided
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        close_after_plotting = True  # Close figure after plotting if axes are created internally
    else:
        close_after_plotting = False

    if cmap in ['red', 'green', 'cyan']:
        cmap = color_dict[cmap]

    vmin, vmax = get_auto_contrast(original_image)

    # Plot original image
    axes[0].imshow(original_image, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot labeled image
    axes[1].imshow(labeled_image, cmap='viridis')
    axes[1].set_title('Labeled Image')
    axes[1].axis('off')

    # Plot inference image
    axes[2].imshow(inference_image, cmap='viridis')
    axes[2].set_title('Inference Image')
    axes[2].axis('off')

    # Set the title if provided
    if title:
        plt.suptitle(title, fontsize=16)

    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=400)

    # Show the plot
    if close_after_plotting:
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_multiple_comparisons(
        original_images: np.ndarray,
        labeled_images: np.ndarray,
        inference_images: np.ndarray,
        cmaps: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot multiple comparisons of original, labeled, and inference images, stacked vertically.

    Parameters
    ----------
    original_images : np.ndarray
        List of original images, each with shape (Y, X).
    labeled_images : np.ndarray
        List of labeled images, each with shape (Y, X).
    inference_images : np.ndarray
        List of inference images, each with shape (Y, X).
    cmaps : Optional[List[str]], default=None
        List of colors to use for the original_images. If None, a default list of colors will be used.
    title : Optional[str], default=None
        Title for the plot.
    save_path : Optional[str], default=None
        If provided, saves the figure to the specified file path.

    Returns
    -------
    Optional[plt.Figure]
        Returns the figure object if neither saving nor showing, otherwise returns None.
    """
    if len(original_images) != len(inference_images) != len(labeled_images):
        raise ValueError("Number of images must match number of labels and inference images.")

    if len(original_images) > 48:
        raise ValueError("Number of images must be less than or equal to 48 to avoid overflow.")

    n_rows = len(original_images)
    # Create the figure with 3*n_rows subplots (3 per row: original, label, and inference)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows), squeeze=False)

    for i in range(n_rows):
        plot_inference_comparison(
            original_image=original_images[i],
            labeled_image=labeled_images[i],
            inference_image=inference_images[i],
            cmap=cmaps[i],
            axes=axes[i],
            title=f'Comparison {i + 1}'
        )

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    # If neither saving nor showing, return the figure object
    return fig

