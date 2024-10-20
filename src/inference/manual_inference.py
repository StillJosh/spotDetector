# manual_inference.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 18.10.24
import torch

import datasets
from inference import inferencer
from inference.load_model_utils import load_pretrained_model
from utils import tools
import plotting as pl

config = {
    'model_name': 'DeepSpotNet',
    'experiment_name': '934313232588188684',
    'run_id': '0cf58c08a9044a7bb1f1189d6e39abcd',
    'ml_flow_path': '../mlruns',
    'model_weights': 'best_model.pth',
    # 'input_size': (256, 256),
}

device = tools.get_device()
model = load_pretrained_model(**config)

ds = datasets.DeepSpotDataset(data_dir='../data/val')
dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
inf = inferencer.Inferencer(model, torch.device('mps'))
images, labels = next(iter(dl))

inf.plot_inference_comparison(images)

