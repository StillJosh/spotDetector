# tools.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 18.10.24
from pathlib import Path
from typing import Optional

import torch
from torch.nn import Module
import wandb  # Import W&B


class CheckpointSaver:
    """
    A class responsible for saving the best model checkpoints during training.

    Attributes:
    ----------
    checkpoint_dir : Path
        The directory where checkpoints and model artifacts will be saved.
    mode : str
        The mode to monitor ('min' or 'max') for saving the best model based on the monitored metric.
    save_best_only : bool
        If True, only the best model is saved based on the monitored metric.
    best_metric : float or None
        The best monitored metric value encountered so far during training.
    """

    def __init__(self, checkpoint_dir: Path, config: dict) -> None:
        """
        Initializes the CheckpointSaver with the given directory and configuration.

        Parameters:
        ----------
        checkpoint_dir : Path
            The directory where model checkpoints will be saved.
        config : dict
            The configuration dictionary containing checkpoint settings.
        """

        self.checkpoint_dir = checkpoint_dir
        self.mode = config['checkpoint']['mode']
        self.save_best_only = config['checkpoint']['save_best_only']
        self.best_metric = None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: Module, metric_to_monitor: float) -> bool:
        """
        Saves the model checkpoint if the current metric is the best observed so far.

        Parameters:
        ----------
        model : Module
            The model to save.
        metric_to_monitor : float
            The current value of the metric to monitor.

        Returns:
        -------

        """

        if self.best_metric is None or (
                self.mode == 'min' and metric_to_monitor < self.best_metric
        ) or (
                self.mode == 'max' and metric_to_monitor > self.best_metric
        ):
            self.best_metric = metric_to_monitor

            # Save the model state_dict
            save_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(model.state_dict(), save_path)

            # Log the model checkpoint as an artifact with W&B
            wandb.save(str(save_path))



class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.

    Parameters
    ----------
    patience : int
        How long to wait after last time validation metric improved.
    verbose : bool
        If True, prints a message for each validation metric improvement.
    delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    mode : str
        'min' or 'max' to decide whether to look for decreasing or increasing metric.

    Attributes
    ----------
    counter : int
        Counts how many times validation metric has not improved.
    best_score : Optional[float]
        Best score achieved so far.
    early_stop : bool
        Whether to stop training early.
    """

    def __init__(
            self, patience: int = 7, verbose: bool = False, delta: float = 0.0, mode: str = 'min'
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, metric: float):
        if self.best_score is None:
            self.best_score = metric
            return

        if self.mode == 'min':
            is_improvement = metric < self.best_score - self.delta
        else:
            is_improvement = metric > self.best_score + self.delta

        if is_improvement:
            self.best_score = metric
            self.counter = 0
            if self.verbose:
                print(f'Validation metric improved to {metric:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


def get_device():
    """
    Get gpu if possible. Otherwise, use cpu.

    Returns
    -------
    torch.device
        The device to use.
    """

    return (
        torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
