# train.py
# Description: Training script for the model.
# Author: Joshua Stiller
# Date: 16.10.24

import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import wandb  # Import Weights and Biases
from tqdm import tqdm

from inference.inferencer import Inferencer
from utils import training_setup as ts
from utils.config_parser import load_config
from utils.logging import logger
from utils.tools import CheckpointSaver, EarlyStopping, get_device


def main(config: Dict[str, Any]):
    # Device management
    # Setup training components

    model = ts.get_model(config)
    train_loader, val_loader = ts.get_dataloaders(config)
    criterion = ts.get_loss_function(config)
    optimizer = ts.get_optimizer(config, model)
    scheduler = ts.get_scheduler(config, optimizer)

    # Early stopping
    early_stopping = EarlyStopping(
        patience=10, verbose=True, mode=config['checkpoint']['mode']
    )

    # Checkpoint directory
    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_saver = CheckpointSaver(checkpoint_dir, config)

    # Inferencer for plotting
    inf = Inferencer(model, *next(iter(val_loader)), device)

    # Log parameters
    wandb.config.update(config['training'])
    wandb.config.update(config['model'])

    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        running_loss = 0.0

        logger.info(f"Epoch [{epoch + 1}/{config['training']['epochs']}] - Training started")

        with tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{config['training']['epochs']}] - Training",
                  file=sys.stdout, ascii=True, ncols=80, mininterval=30) as t:
            for inputs, labels, _ in t:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                t.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader.dataset)
        wandb.log({'train_loss': epoch_loss, 'epoch': epoch})  # Log train loss
        logger.info(f"Epoch [{epoch + 1}/{config['training']['epochs']}] - Training loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0

        logger.info(f"Epoch [{epoch + 1}/{config['training']['epochs']}] - Validation started")

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{config['training']['epochs']}] - Validation",
                      file=sys.stdout, ascii=True, ncols=80) as t:
                for inputs, labels, _ in t:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * inputs.size(0)
                    t.set_postfix({"Val Loss": f"{loss.item():.4f}"})

        val_loss = val_running_loss / len(val_loader.dataset)
        wandb.log({'val_loss': val_loss, 'epoch': epoch})  # Log validation loss
        logger.info(f"Epoch [{epoch + 1}/{config['training']['epochs']}] - Validation loss: {val_loss:.4f}")

        # Learning rate scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

        # Checkpoint saving
        checkpoint_saver.save(model, val_loss)

        # Log inference comparison figure
        fig = inf.plot_inference_comparison()
        wandb.log({"inference_comparison": wandb.Image(fig)})  # Log figure to W&B
        logger.info(f"Logged inference comparison figure for epoch {epoch + 1}")

    logger.info("Training completed.")
    wandb.finish()  # Finish W&B run


if __name__ == '__main__':

    config = load_config(Path(__file__).parent / 'config' / 'config.yaml')

    if config['debug']:
        os.environ['WANDB_MODE'] = 'offline'

    # W&B setup
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(project='spotDetector',
               config={**config['training'], **config['model'], **config['data']})

    main(config)
