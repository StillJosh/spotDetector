# utils.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 19.10.24

import numpy as np
import torch


class ClipOutliers(torch.Module):
    """
    Clips outliers of the image data between specified quantile.

    Parameters
    ----------
    img : np.ndarray
        The input image array.
    min_val : int
        The minimum value of the output normalized data.
    max_val : int
        The maximum value of the output normalized data.
    q_low : float, optional
        The lower quantile for clipping, by default 0.005.
    q_high : float, optional
        The upper percentile for clipping, by default 0.995.

    """

    def __init__(self, img, min_val, max_val, lower_percentile=0.5, upper_percentile=99.5):
        self.img = img
        self.min_val = min_val
        self.max_val = max_val
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def __call__(self):
        """
        Clips outliers of the image data between specified quantile.

        Returns
        -------
        np.ndarray
            The clipped image array.
        """

        low = np.percentile(self.img, self.lower_percentile)
        high = np.percentile(self.img, self.upper_percentile)
        self.img = np.clip(self.img, low, high)
        self.img = np.clip(self.img, self.min_val, self.max_val)

        return self.img
