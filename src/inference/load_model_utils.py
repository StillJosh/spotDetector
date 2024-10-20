# load_model_utils.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 18.10.24


from pathlib import Path
from typing import Union, Optional

import torch

from src import models


def load_pretrained_model(
        model_name: str,
        experiment_name: str,
        run_id: str,
        ml_flow_path: Union[Path, str] = '../mlruns',
        model_weights: str = 'best_model.pth',
        model_kwargs: Optional[dict] = None,
) -> torch.nn.Module:
    """
    Load a pretrained model from MLflow.

    Parameters
    ----------
    model_name : str
        The name of the model to load.
    experiment_name : str
        The name of the experiment containing the model.
    run_id : str
        The ID of the run containing the model.
    ml_flow_path : Union[Path, str]
        The path to the MLflow directory.
    model_weights : str
        The name of the model weights file.
    model_kwargs : Optional[dict]
        Additional keyword arguments to pass to the model constructor.

    Returns
    -------
    torch.Module
        The pretrained model.
    """
    model_kwargs = model_kwargs or {}

    model_path = Path(ml_flow_path) / experiment_name / run_id / 'artifacts' / model_weights

    model = getattr(models, model_name)(**model_kwargs)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model
