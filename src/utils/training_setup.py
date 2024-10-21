# training_setup.py
# Description: Setup the model, loss function, optimizer, and scheduler.
# Author: Joshua Stiller
# Date: 16.10.24
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import boto3
import datasets as ds
import models
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import utils.losses as cl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


def setup_training_components(
        config: Dict[str, Any], device: torch.device, debug: bool = False
) -> Tuple[nn.Module, nn.Module, Optimizer, _LRScheduler]:
    """
    Set up the model, loss function, optimizer, and scheduler.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    device : torch.device
        Device to run the model on.
    debug : bool
        Whether to run the dataset in debug mode. If True, only two samples will be loaded.

    Returns
    -------
    Tuple[nn.Module, DataLoader, DataLoader, nn.Module, Optimizer, _LRScheduler]
        The model, train dataloader, validation dataloader, loss function, optimizer, and scheduler.
    """

    # Model
    model = get_model(config['model'])
    model = model.to(device)

    def get_cuda_device_ids():
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            return list(zip(range(device_count), device_names))
        else:
            return []

    # Example usage
    cuda_devices = get_cuda_device_ids()
    print(f'Cuda Devices: {cuda_devices}')

    # if device.type == 'cuda' and len(config['device']['gpu_ids']) > 1:
    #    model = nn.DataParallel(model, device_ids=config['device']['gpu_ids'])

    # Dataset
    if config['data']['dataset'] == 'deepspot':
        train_loader, val_loader = get_dataset(ds.DeepSpotDataset, config, debug)
    elif config['data']['dataset'] == 'deepspot_s3':
        train_loader, val_loader = get_dataset(ds.DeepSpotDatasetS3, config, debug)

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


def get_dataset(dataset: Callable, config: Dict[str, Any], debug: bool) -> Tuple[DataLoader, DataLoader]:
    """
    Loads a dataset from the datasets module according to information in config.

    Parameters
    ----------
    dataset: Callable
        A class from the datasets module.
    config : Dict[str, Any]
        Configuration dictionary.
    debug : bool
        Whether to run the dataset in debug mode. If True, only two samples will be loaded.
    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation data loaders.
    """

    root_dir = Path(config['data']['root_dir'])
    if config['data']['dataset'].endswith('_s3'):
        metadata = pl.read_csv(
            boto3.client('s3')
            .get_object(Bucket='tunamlbucket', Key='spotDetection/metadata_full.csv')['Body']
            .read())
    else:
        metadata = pl.read_csv(root_dir / 'metadata_full.csv')

    metadata = metadata.filter(pl.col('bit_depth') == config['data']['bit_depth'])


    # Data loaders
    train_dataset = dataset(
        data_dir=Path(config['data']['root_dir']),
        metadata=metadata.filter(pl.col('phase') == 'train'),
        input_size=tuple(config['data']['input_size']),
        debug=debug
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    val_dataset = dataset(
        data_dir=Path(config['data']['root_dir']),
        metadata=metadata.filter(pl.col('phase') == 'val'),
        input_size=tuple(config['data']['input_size']),
        debug=debug
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    return train_loader, val_loader


