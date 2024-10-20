# inference_plotting.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 18.10.24


from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_inference_comparison(
        original_image: np.ndarray,
        inference_image: np.ndarray,
        title: Optional[str] = None,
        axes: Optional[Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]] = None,
        save_path: Optional[str] = None
) -> None:
    """
    Plot the original image and its corresponding inference image side by side.

    Parameters
    ----------
    original_image : np.ndarray
        The original image with shape (Y, X).
    inference_image : np.ndarray
        The inference image with shape (Y, X).
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
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        close_after_plotting = True  # Close figure after plotting if axes are created internally
    else:
        close_after_plotting = False

    # Ensure axes is a tuple if not already
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes, axes]

    # Plot original image
    axes[0].imshow(original_image, cmap='viridis')
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Remove axis ticks

    # Plot inference image
    axes[1].imshow(inference_image, cmap='viridis')
    axes[1].set_title('Inference Image')
    axes[1].axis('off')  # Remove axis ticks

    # Set the title if provided
    if title:
        plt.suptitle(title, fontsize=16)

    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    if close_after_plotting:
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_multiple_comparisons(
        original_images: np.ndarray,
        inference_images: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot multiple comparisons of original and inference images, stacked vertically.

    Parameters
    ----------
    original_images : np.ndarray
        List of original images, each with shape (Y, X).
    inference_images : np.ndarray
        List of inference images, each with shape (Y, X).
    title : Optional[str], default=None
        Title for the plot.
    save_path : Optional[str], default=None
        If provided, saves the figure to the specified file path.
    show : bool, default=False
        If True, shows the plot.

    Returns
    -------
    Optional[plt.Figure]
        Returns the figure object if neither saving nor showing, otherwise returns None.
    """
    if len(original_images) != len(inference_images):
        raise ValueError("Number of original images must match number of inference images.")

    n_rows = len(original_images)

    # Create the figure with 2*n_rows subplots (2 per row: original and inference)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows))

    # If only one row, make axes a list to ensure consistency
    if n_rows == 1:
        axes = [axes]

    for i in range(n_rows):
        plot_inference_comparison(
            original_image=original_images[i],
            inference_image=inference_images[i],
            axes=axes[i],
            title=f'Comparison {i + 1}'
        )

    # Set the overall title
    if title:
        fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the figure if requested
    if show:
        plt.show()
        plt.close(fig)
        return None

    # If neither saving nor showing, return the figure object
    return fig

