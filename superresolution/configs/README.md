### Introduction
This project uses [Hydra](https://hydra.cc/) (developed by Meta) as its experiment configuration manager. Hydra composes nested YAML files into a single runtime config object and supports command-line overrides for quick experimentation.

Install:
```
conda install conda-forge::hydra-core=1.3.2 conda-forge::omegaconf=2.3.0
```

### Layout
The `configs/` directory is flat — one YAML file per experiment plus a shared base:

```
configs/
  default.yaml       # shared baseline for every experiment
  cnn.yaml           # each model file extends default and sets model
  closure_cnn.yaml
  upsample_cnn.yaml
  fno.yaml
  gnn.yaml
```

`default.yaml` contains the download, preprocess, and train blocks that every experiment uses (and the `raw_data_dir` / `processed_data_dir` / `checkpoints_dir` knobs). It deliberately does not set `model` — a runnable experiment must specify one.

Each model-specific yaml declares a defaults list that pulls in `default.yaml`, then adds its own `model` block:

```yaml
# cnn.yaml
defaults:
  - default
  - _self_

model:
  name: cnn
  params:
    _target_: superresolution.models.cnn.SuperResolutionCNN
    hidden_channels: 32
    num_blocks: 4
```

The `_target_` field tells `hydra.utils.instantiate()` which class to construct; the rest of `params` becomes constructor kwargs. `model.name` is a plain string used by `preprocess.py` to pick the right `make_training_pair` function.

### Structured Configs (Type Safety)
Dataclasses in `config.py` define the schema for the composed config. `SuperResolutionConfig` ties them together:

```python
@dataclass
class SuperResolutionConfig:
    download: DownloadConfig = MISSING
    preprocess: PreprocessConfig = MISSING
    train: TrainConfig = MISSING
    model: Any = MISSING
    raw_data_dir: Optional[str] = None
    processed_data_dir: Optional[str] = None
```

Each entry-point script (`train.py`, `preprocess.py`, `download.py`) registers this schema with Hydra under the name `base_config`:

```python
from hydra.core.config_store import ConfigStore
from .config import SuperResolutionConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=SuperResolutionConfig)
```

`default.yaml` then pulls in `base_config` via its own defaults list:
```yaml
defaults:
  - base_config
  - _self_
```

Any yaml that extends `default.yaml` transitively inherits the schema, so type validation applies to all model configs without having to repeat the reference per file.

### Running
Each entry-point script uses `@hydra.main` with `config_name="cnn"` as the default root, so running with no arguments uses `cnn.yaml`. Pass `--config-name=<model>` to select a different experiment:

```bash
python -m superresolution.train                           # cnn (default)
python -m superresolution.train --config-name=gnn         # gnn
python -m superresolution.preprocess --config-name=fno    # fno
```

Override individual fields on top of the selected config with standard Hydra syntax:

```bash
python -m superresolution.train --config-name=cnn train.epochs=500 train.batch_size=4
python -m superresolution.download raw_data_dir=/path/to/raw
```

**Working dir caveat**: Hydra automatically `cd`s into a timestamped `outputs/` directory at startup so each run's logs and checkpoints are isolated. Inside any `@hydra.main`-decorated function, always resolve file paths with `Path(__file__).resolve()` — relative paths will point to the wrong location.
