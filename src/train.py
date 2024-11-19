# train.py
# Description: Training script for the model.
# Author: Joshua Stiller
# Date: 16.10.24

import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import wandb  # Import Weights and Biases
from tqdm import tqdm

from plotting.inference_plotting import plot_multiple_comparisons
from utils import training_setup as ts
from utils.config_parser import load_config
from utils.logger import logger
from utils.tools import CheckpointSaver, EarlyStopping


def get_images_for_plotting(model, dataloader, device, num_images=48):
    model.eval()
    num_images = min(num_images, len(dataloader.dataset))
    image_res, label_res, pred_res, colors = [], [], [], []

    for images, labels, idxs in dataloader:
        if len(image_res) >= num_images:
            break

        with torch.no_grad():
            predictions = model(images.to(device)).cpu()

        batch_size = images.shape[0]
        for i in range(batch_size):
            if images[i].ndim == 4:  # For 3D images, extract 3 slices per image
                depth = images[i].shape[2]
                slide_index = np.random.choice(np.arange(1, depth - 1), size=1)
                img = images[i][0, slide_index - 1:slide_index + 2, :, :]  # Extract slices (Channels, 3, Height, Width)
                lab = labels[i][0, slide_index - 1:slide_index + 2, :, :]
                pred = predictions[i][0, slide_index - 1:slide_index + 2, :, :]
            else:
                img = images[i]
                lab = labels[i]
                pred = predictions[i]

            image_res.extend([im for im in img])
            label_res.extend([lab for lab in lab])
            pred_res.extend([pr for pr in pred])
            colors.extend(dataloader.dataset.metadata[idxs.cpu().numpy(), 'color'])

    return image_res[:num_images], label_res[:num_images], pred_res[:num_images], colors[:num_images]


def main(config: Dict[str, Any]):
    device = config['device']

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

    images, labels, preds, colors = get_images_for_plotting(model, val_loader, device)
    fig = plot_multiple_comparisons(images, labels, preds, cmaps=colors, title="Inference Comparison")
    wandb.log({"inference_comparison": wandb.Image(fig), 'epoch': 0})

    for epoch in range(config['training']['epochs']):
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
        wandb.log({'val_loss': val_loss, 'epoch': epoch + 1})  # Log validation loss
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
        images, labels, preds, colors = get_images_for_plotting(model, val_loader, device)
        fig = plot_multiple_comparisons(images, labels, preds, cmaps=colors, title="Inference Comparison")
        wandb.log({"inference_comparison": wandb.Image(fig), 'epoch': epoch + 1})
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
