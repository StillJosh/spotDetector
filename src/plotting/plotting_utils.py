# plotting_utils.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 05.11.24

import numpy as np


def get_auto_contrast(page: np.ndarray) -> tuple[float, float]:
    """
    Implements Fiji-like auto-contrasting

    Parameters
    ----------
    page : np.ndarray
        Image data. Should usually be a 2D array (Y, X).
    """

    if page.size == 0:
        raise ValueError('Empty page')

    base_val = 5000

    hist, bins = np.histogram(page, bins=256)

    # Find the first and last bin with values
    thresh = np.prod(page.shape) / base_val
    first_bin = np.argmax(hist > thresh)
    last_bin = 255 - np.argmax(hist[::-1] > thresh)

    min_contrast, max_contrast = bins[[first_bin, last_bin]]

    return min_contrast, max_contrast
