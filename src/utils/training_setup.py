# training_setup.py
# Description: Setup the model, loss function, optimizer, and scheduler.
# Author: Joshua Stiller
# Date: 16.10.24
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import datasets as ds
import models
import utils.losses as cl


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

    if config['model']['name'] == 'deepspot':
        model = models.DeepSpotNet(
            input_channels=config.get('input_channels', 1),
            dropout_rate=config.get('dropout_rate', 0.2),
            conv_block4_filters=config.get('conv_block4_filters', 128),
            identity_block_filters=config.get('identity_block_filters', 128),
        )
    elif config['model']['name'] == 'deepspot3D':
        model = models.DeepSpotNet3D(
            input_channels=config.get('input_channels', 1),
            dropout_rate=config.get('dropout_rate', 0.2),
            conv_block4_filters=config.get('conv_block4_filters', 128),
            identity_block_filters=config.get('identity_block_filters', 128),
        )
    else:
        raise ValueError(f"Model {config['model']['name']} not recognized.")

    if config['model']['pretrained'] != 'None' and config['model']['pretrained'] != '':
        pretrained_path = Path(config['data']['root_dir']).joinpath('pretrained_models', config['model']['pretrained'])
        model.load_state_dict(torch.load(pretrained_path, weights_only=True))

    return model


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Loads train and val data (incl. metadata) and returns the corresponding data loaders.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    debug : bool
        Whether to run the dataset in debug mode. If True, only two samples will be loaded.
    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation data loaders.
    """

    datasets = {'deepSpot': ds.DeepSpotDataset,
                # 'deepSpot3D': ds.DeepSpot3DDataset
                }

    dataset = datasets[config['data']['dataset']]
    data_dir = Path(config['data']['root_dir']) / config['data']['data_dir']

    # Load and filter metadata for training and validation
    metadata_train, metadata_val = _load_metadata(config, data_dir)

    # Data loaders
    train_dataset = dataset(
        data_file=data_dir / config['data']['train_file'],
        metadata=metadata_train,
        input_size=tuple(config['data']['input_size']),
        augmentations=config['data']['augmentations'],
        is_train=True,
        debug=config['debug']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    val_dataset = dataset(
        data_file=data_dir / config['data']['val_file'],
        metadata=metadata_val,
        input_size=tuple(config['data']['input_size']),
        augmentations=config['data']['augmentations'],
        is_train=False,
        debug=config['debug']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size_val'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    return train_loader, val_loader


def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """
    Initialize and return the loss function based on configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Loss function configuration parameters.

    Returns
    -------
    nn.Module
        Initialized loss function.
    """

    if config['training']['loss'] == 'deepspot':
        criterion = cl.DeepSpotLoss()
    else:
        raise ValueError(f"Loss function {config['training']['loss']} not recognized.")

    return criterion


def get_optimizer(config: Dict[str, Any], model: nn.Module) -> Optimizer:
    """
    Initialize and return the optimizer based on configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Optimizer configuration parameters.
    model : nn.Module
        Model to optimize.

    Returns
    -------
    Optimizer
        Initialized optimizer.
    """

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

    return optimizer


def get_scheduler(config: Dict[str, Any], optimizer: Optimizer) -> _LRScheduler:
    """
    Initialize and return the scheduler based on configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Scheduler configuration parameters.
    optimizer : Optimizer
        Optimizer to schedule.

    Returns
    -------
    _LRScheduler
        Initialized scheduler.
    """

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

    return scheduler


def _load_metadata(config: Dict[str, Any], data_dir: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load and filter metadata for training and validation.

    Parameters
    ----------
    config: Dict[str, Any]
        Configuration dictionary.
    data_dir: Path
        Path to the data directory.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        Filtered metadata for training and validation.
    """

    # Load full metadata files
    metadata_train = pl.read_csv(data_dir / 'metadata_train.csv')
    metadata_val = pl.read_csv(data_dir / 'metadata_val.csv')

    # Filter metadata based on training config
    metadata_train = metadata_train.filter((pl.col('bit_depth') == config['data']['bit_depth']) &
                                           (pl.col('mode') == config['data']['mode']))
    metadata_val = metadata_val.filter((pl.col('bit_depth') == config['data']['bit_depth']) &
                                       (pl.col('mode') == config['data']['mode']))
    for filter_name in config['data']['filter_name']:
        metadata_train = metadata_train.filter(pl.col('image_name').str.contains(filter_name))
        metadata_val = metadata_val.filter(pl.col('image_name').str.contains(filter_name))

    return metadata_train, metadata_val
