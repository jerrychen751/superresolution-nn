"""
Load saved model weights and run predictions on inference_data_dir (defaults to
the test split). Writes one prediction_t{XXXX}.npy per input file to outputs_dir.

Per-model input/output plumbing lives in each model file's build_inference_fn
factory; this file only handles the shared wiring (weights, I/O, dispatch).
"""

import numpy as np
import torch
import hydra
import hydra.utils
from pathlib import Path


def _get_build_inference_fn(model_name: str):
    """
    Return each model's build_inference_fn factory. Same dispatch pattern as
    train.py's _get_data_classes and preprocess.py's make_training_pair import.
    """
    if model_name == "cnn":
        from .models.cnn import build_inference_fn
    elif model_name == "upsample_cnn":
        from .models.upsample_cnn import build_inference_fn
    elif model_name == "closure_cnn":
        from .models.closure_cnn import build_inference_fn
    elif model_name == "fno":
        from .models.fno import build_inference_fn
    elif model_name == "gnn":
        from .models.gnn import build_inference_fn
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return build_inference_fn


@hydra.main(version_base=None, config_path="configs", config_name="cnn")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model.params).to(device)
    state_dict = torch.load(
        Path(cfg.weights_dir) / "weights.pth",
        map_location=device, weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    inference_dir = Path(cfg.inference_data_dir)
    input_fps = sorted(inference_dir.glob("input_t*.npy"))
    if not input_fps:
        print(f"No input files found in {inference_dir}")
        return

    output_dir = Path(cfg.outputs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_shape = np.load(input_fps[0]).shape
    predict = _get_build_inference_fn(cfg.model.name)(model, sample_shape, device)

    with torch.inference_mode():
        for input_fp in input_fps:
            t_str = input_fp.stem.split("_t")[1]
            input_array = np.load(input_fp).astype(np.float32)
            pred = predict(input_array)
            np.save(output_dir / f"prediction_t{t_str}.npy", pred)
            print(f"[inference] {input_fp.name} -> prediction_t{t_str}.npy  shape={pred.shape}")

    print(f"Saved {len(input_fps)} predictions to {output_dir}")


if __name__ == "__main__":
    main()
