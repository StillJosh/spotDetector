# training_setup.py
# Description: Setup the model, loss function, optimizer, and scheduler.
# Author: Joshua Stiller
# Date: 16.10.24

from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import models
import utils.losses as cl
import datasets as ds


def setup_training_components(
        config: Dict[str, Any], device: torch.device
) -> Tuple[nn.Module, nn.Module, Optimizer, _LRScheduler]:
    """
    Set up the model, loss function, optimizer, and scheduler.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    device : torch.device
        Device to run the model on.

    Returns
    -------
    Tuple[nn.Module, DataLoader, DataLoader, nn.Module, Optimizer, _LRScheduler]
        The model, train dataloader, validation dataloader, loss function, optimizer, and scheduler.
    """

    # Model
    model = get_model(config['model'])
    model = model.to(device)
    if device.type == 'cuda' and len(config['device']['gpu_ids']) > 1:
        model = nn.DataParallel(model, device_ids=config['device']['gpu_ids'])

    # Dataset
    if config['data']['dataset'] == 'deepspot':
        train_loader, val_loader = get_dataset(ds.DeepSpotDataset, config)

    # Loss function
    if config['training']['loss'] == 'deepspot':
        criterion = cl.DeepSpotLoss()

    # Optimizer
    optimizer_name = config['training']['optimizer'].lower()
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=config['training'].get('momentum', 0.9),
        )
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not recognized.")

    # Scheduler
    scheduler_name = config['training']['scheduler'].lower()
    scheduler_params = config['training']['scheduler_params']
    if scheduler_name == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params['step_size'],
            gamma=scheduler_params['gamma'],
        )
    elif scheduler_name == 'reducelronplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_params['factor'],
            patience=scheduler_params['patience'],
            threshold=scheduler_params.get('threshold', 0.0001),
        )
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' not recognized.")

    return model, train_loader, val_loader, criterion, optimizer, scheduler


def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Initialize and return the model based on configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Model configuration parameters.

    Returns
    -------
    nn.Module
        Initialized model.
    """

    if config['name'] == 'deepspot':
        model = models.DeepSpotNet(
            input_channels=config.get('input_channels', 1),
            dropout_rate=config.get('dropout_rate', 0.2),
            conv_block4_filters=config.get('conv_block4_filters', 128),
            identity_block_filters=config.get('identity_block_filters', 128),
        )
    else:
        raise ValueError(f"Model {config['name']} not recognized.")
    return model


def get_dataset(dataset: Callable, config: Dict[str, Any]):
    """
    Loads a dataset from the datasets module according to information in config.

    Parameters
    ----------
    dataset: Callable
        A class from the datasets module.
    config : Dict[str, Any]
        Configuration dictionary.
    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation data loaders.
    """

    # Data loaders
    train_dataset = dataset(
        config['data']['train_dir'], tuple(config['data']['input_size'])
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
    )

    val_dataset = dataset(
        config['data']['val_dir'], tuple(config['data']['input_size'])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
    )

    return train_loader, val_loader


