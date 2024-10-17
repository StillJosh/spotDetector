# checkpoint_saver.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 17.10.24

# utils/checkpoint_utils.py
import torch
import mlflow
from pathlib import Path

from torch import Module


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

            save_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(model.state_dict(), save_path)
            mlflow.log_artifact(str(save_path))

            # Export to TorchScript
            scripted_model = torch.jit.script(model)
            script_save_path = self.checkpoint_dir / 'best_model_scripted.pt'
            scripted_model.save(str(script_save_path))
            mlflow.log_artifact(str(script_save_path))
