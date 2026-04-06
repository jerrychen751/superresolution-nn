"""
This file loads saved model weights and uses it for model inference.
"""

import torch
import hydra.utils
from hydra import compose, initialize_config_dir
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_DIR / "configs"

with initialize_config_dir(str(CONFIG_DIR), version_base=None):
    cfg = compose(config_name="config")

# Instantiate model from config: hydra resolves _target_ to the class and passes the remaining keys as constructor kwargs
model = hydra.utils.instantiate(cfg.model.params)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load saved weights
state_dict_dir = cfg.weights_dir if cfg.weights_dir else PROJECT_DIR / "weights" / cfg.model.name
state_dict = torch.load(state_dict_dir / "weights.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

output_dir = cfg.outputs_dir if cfg.outputs_dir else PROJECT_DIR / "outputs" / cfg.model.name
with torch.inference_mode():
