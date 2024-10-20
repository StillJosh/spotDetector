# train.py
# Description: Training script for the model.
# Author: Joshua Stiller
# Date: 16.10.24

from pathlib import Path
from typing import Any, Dict

import mlflow
import torch
from tqdm import tqdm
import argparse

from src.utils.tools import CheckpointSaver, EarlyStopping, get_device
from src.utils.config_parser import load_config
from src.utils.training_setup import setup_training_components
from src.inference.inferencer import Inferencer


def main(config: Dict[str, Any]):
    # Device management

    device = get_device()

    # Setup training components
    model, train_loader, val_loader, criterion, optimizer, scheduler = setup_training_components(config, device, debug=debug)

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

            update_str = f"Epoch {epoch + 1}/{config['training']['epochs']} - Training"
            with tqdm(train_loader, desc=update_str, leave=False) as t:
                for inputs, labels in t:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    t.set_postfix({"Loss": running_loss / len(train_loader)})

            epoch_loss = running_loss / len(train_loader.dataset)
            mlflow.log_metric('train_loss', epoch_loss, step=epoch)

            # Validation phase
            model.eval()
            val_running_loss = 0.0

            with torch.no_grad():
                update_str = f"Epoch {epoch + 1}/{config['training']['epochs']} - Validation"
                with tqdm(train_loader, desc=update_str, leave=False) as t:
                    for inputs, labels in t:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        val_running_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        t.set_postfix({"Loss": running_loss / len(val_loader)})

            val_loss = val_running_loss / len(val_loader.dataset)
            mlflow.log_metric('val_loss', val_loss, step=epoch)

            # LR Scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # Early stopping check
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # Checkpoint saving
            checkpoint_saver.save(model, val_loss)

            print(
                f"\nEpoch [{epoch + 1}/{config['training']['epochs']}] - "
                f"Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - "
            )

            fig = inf.plot_inference_comparison()
            mlflow.log_figure(fig, f"inference_comparison_{epoch}.png")

    print("Training completed.")


debug = False

if __name__ == '__main__':
    config = load_config(Path(__file__).parent / 'config' / 'config.yaml')

    # Attach input data if on AzureML
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str)
    args = parser.parse_args()
    if args.input_data is not None:
        config['data']['root_dir'] = args.input_data
    main(config)
