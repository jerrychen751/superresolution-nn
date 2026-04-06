"""
Module contains dataclass definitions for expected configs for the ML experiment.

Hydra handles the loading of configs.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

@dataclass
class DownloadConfig:
    jhtdb_dataset: str
    jhtdb_cache_dir: str
    nx: int
    ny: int
    nz: int
    num_cubes: int
    max_timesteps: int

    jhtdb_token: str = "edu.gatech.jerrychen-6b7455a7"

@dataclass
class PreprocessConfig:
    sigma: float # standard deviation of Gaussian filter PDF
    downsample_step: int # stride used when downsampling
    mode: str = MISSING  # set via interpolation from root config's 'model' field

    spline_interpolation_order: int = 3

@dataclass
class TrainConfig:
    # Overall train/test settings
    epochs: int
    train_ratio: float

    # Learning rate scheduler (cosine annealing)
    learning_rate: float
    eta_min: float

    # DataLoader
    batch_size: int
    num_workers: int

@dataclass
class SuperResolutionConfig:
    download: DownloadConfig = MISSING
    preprocess: PreprocessConfig = MISSING
    train: TrainConfig = MISSING
    model: Any = MISSING  # loaded from configs/model/ config group
    raw_data_dir: Optional[str] = None
    processed_data_dir: Optional[str] = None
    checkpoints_dir: Optional[str] = None
    weights_dir: Optional[str] = None
    outputs_dir: Optional[str] = None

