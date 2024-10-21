# train.py
# Description: Training script for the model.
# Author: Joshua Stiller
# Date: 16.10.24

import argparse
import sys
from pathlib import Path
from typing import Any, Dict
import mlflow
import torch
from tqdm import tqdm

from inference.inferencer import Inferencer
from utils.config_parser import load_config
from utils.tools import CheckpointSaver, EarlyStopping, get_device
from utils.training_setup import setup_training_components
from utils.logging import logger

def main(config: Dict[str, Any]):
    # Device management
    device = get_device()

    # Setup training components
    model, train_loader, val_loader, criterion, optimizer, scheduler = setup_training_components(config, device,
                                                                                                 debug=debug)

    # MLflow setup
    mlflow.set_tracking_uri(config['logging']['mlflow_tracking_uri'])
    mlflow.set_experiment(config['logging']['experiment_name'])

    # Early stopping
    early_stopping = EarlyStopping(
        patience=10, verbose=True, mode=config['checkpoint']['mode']
    )

    # Checkpoint directory
    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_saver = CheckpointSaver(checkpoint_dir, config)

    # Inferencer for plotting
    inf = Inferencer(model, next(iter(val_loader))[0], device)

    with mlflow.start_run():
        mlflow.log_params(config['training'])
        mlflow.log_params(config['model'])

        for epoch in range(config['training']['epochs']):
            # Training phase
            model.train()
            running_loss = 0.0

            logger.info(f"Epoch [{epoch + 1}/{config['training']['epochs']}] - Training started")

            with tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{config['training']['epochs']}] - Training",
                      file=sys.stdout, ascii=True, ncols=80) as t:
                for inputs, labels in t:
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
            mlflow.log_metric('train_loss', epoch_loss, step=epoch)
            logger.info(f"Epoch [{epoch + 1}/{config['training']['epochs']}] - Training loss: {epoch_loss:.4f}")

            # Validation phase
            model.eval()
            val_running_loss = 0.0

            logger.info(f"Epoch [{epoch + 1}/{config['training']['epochs']}] - Validation started")

            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{config['training']['epochs']}] - Validation",
                          file=sys.stdout, ascii=True, ncols=80) as t:
                    for inputs, labels in t:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        val_running_loss += loss.item() * inputs.size(0)
                        t.set_postfix({"Val Loss": f"{loss.item():.4f}"})

            val_loss = val_running_loss / len(val_loader.dataset)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
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
            mlflow.log_figure(fig, f"inference_comparison_{epoch}.png")
            logger.info(f"Logged inference comparison figure for epoch {epoch + 1}")

    logger.info("Training completed.")

debug = False

if __name__ == '__main__':
    config = load_config(Path(__file__).parent / 'config' / 'config.yaml')

    # Attach input data if provided
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str)
    args = parser.parse_args()
    if args.input_data is not None:
        config['data']['root_dir'] = args.input_data

    main(config)