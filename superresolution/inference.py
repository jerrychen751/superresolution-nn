"""
This file loads saved model weights and uses it for model inference.
"""

import torch
import hydra
import hydra.utils
from pathlib import Path


@hydra.main(version_base=None, config_path="configs", config_name="cnn")
def main(cfg):
    # Instantiate model from config: hydra resolves _target_ to the class and passes the remaining keys as constructor kwargs
    model = hydra.utils.instantiate(cfg.model.params)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Load saved weights into model
    weights_dir = Path(cfg.weights_dir)
    state_dict = torch.load(weights_dir / "weights.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Prepare input data for model

    # Prepare output dir for model predictions
    output_dir = Path(cfg.outputs_dir)


    model.eval()
    with torch.inference_mode():
        pass


if __name__ == "__main__":
    main()
