# train.py
# Description: Training script for the model.
# Author: Joshua Stiller
# Date: 16.10.24

from pathlib import Path
from typing import Any, Dict

import mlflow
import torch
from tqdm import tqdm

from utils.config_parser import load_config
from utils.early_stopping import EarlyStopping
from utils.training_setup import setup_training_components


def main(config: Dict[str, Any]):
    # Device management
    device = (
        torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if device.type == 'cuda' and len(config['device']['gpu_ids']) > 1:
        torch.cuda.set_device(config['device']['gpu_ids'][0])

    # Setup training components
    model, train_loader, val_loader, criterion, optimizer, scheduler = setup_training_components(config, device)

    # MLflow setup
    mlflow.set_tracking_uri(config['logging']['mlflow_tracking_uri'])
    mlflow.set_experiment(config['logging']['experiment_name'])

    # Early stopping
    early_stopping = EarlyStopping(
        patience=10, verbose=True, mode=config['checkpoint']['mode']
    )

    # Checkpoint directory
    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_metric = None

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
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_running_loss / len(val_loader.dataset)
            val_accuracy = 100 * correct / total
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_accuracy', val_accuracy, step=epoch)

            # Scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # Early stopping
            if config['checkpoint']['monitor_metric'] == 'val_loss':
                metric_to_monitor = val_loss
            elif config['checkpoint']['monitor_metric'] == 'val_accuracy':
                metric_to_monitor = -val_accuracy  # Negative because we want to maximize accuracy
            else:
                raise ValueError("Unsupported monitor metric for early stopping.")

            early_stopping(metric_to_monitor)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # Checkpoint saving
            if best_metric is None:
                best_metric = metric_to_monitor
                is_best = True
            else:
                if config['checkpoint']['mode'] == 'min':
                    is_best = metric_to_monitor < best_metric
                else:
                    is_best = metric_to_monitor > best_metric

            if is_best:
                best_metric = metric_to_monitor
                if config['checkpoint']['save_best_only']:
                    # Save the best model
                    save_path = checkpoint_dir / 'best_model.pth'
                    torch.save(model.state_dict(), save_path)
                    mlflow.log_artifact(str(save_path))

                    # Export to TorchScript
                    scripted_model = torch.jit.script(model)
                    script_save_path = checkpoint_dir / 'best_model_scripted.pt'
                    scripted_model.save(str(script_save_path))
                    mlflow.log_artifact(str(script_save_path))

            print(
                f"Epoch [{epoch + 1}/{config['training']['epochs']}] - "
                f"Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_accuracy:.2f}%"
            )

    print("Training completed.")


if __name__ == '__main__':
    config = load_config('config/config.yaml')
    main(config)
