# inference.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 16.10.24

# inference.py
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms

from utils.training_setup import get_model
from utils.config_parser import load_config


def load_image(image_path: Path, input_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load and preprocess an image.

    Parameters
    ----------
    image_path : Path
        Path to the image.
    input_size : Tuple[int, int]
        Desired input size (height, width).

    Returns
    -------
    torch.Tensor
        Preprocessed image tensor.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def inference(config: Dict[str, Any], image_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(config['model'])
    checkpoint_path = Path(config['checkpoint']['dir']) / 'best_model.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load image
    input_size = tuple(config['data']['input_size'])
    image = load_image(Path(image_path), input_size)
    image = image.to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        print(f'Predicted class: {predicted.item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()

    config = load_config('config/config.yaml')
    inference(config, args.image)
