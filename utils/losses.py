# metrics.py
# Description: Custom losses for model evaluation.
# Author: Joshua Stiller
# Date: 16.10.24

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSpotLoss(nn.Module):
    """
    Custom loss function used in DeepSpot for RNA spot enhancement.

    The loss combines Binary Cross-Entropy (BCE) with Mean Squared Error (MSE)
    to enhance spot intensity and accuracy.

    Parameters
    ----------
    None

    Methods
    -------
    forward(pred, target)
        Computes the combined BCE and MSE loss.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined BCE and MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted output from the network, should have values between 0 and 1.
        target : torch.Tensor
            Ground truth image with same dimensions as `pred`.

        Returns
        -------
        torch.Tensor
            The combined BCE and MSE loss value.
        """
        bce_loss = F.binary_cross_entropy(pred, target)

        # Calculate the maximum values of the predicted and target images
        max_pred = torch.max(pred)
        max_target = torch.max(target)
        mse_loss = F.mse_loss(max_pred, max_target)

        return bce_loss + mse_loss
