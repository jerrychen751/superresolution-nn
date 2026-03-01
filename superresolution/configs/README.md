### Introduction
Lots of industry ML use Hydra (developed by Meta) as the standard for an experiment configuration manager, which utilizes YAML files to specify things that may vary across experiments. For example, the number of training data points that we want, or the number of epochs that we want the training to go on for, or even the dataset that we want to be loading from the JHU Turbulence Database.

Use the following command for installation:

```
conda install conda-forge::hydra-core=1.3.2 conda-forge::omegaconf=2.3.0
```

### Config Setup
`configs/` contains a root YAML file, as well as multiple subfolders. The names of the subfolders are based on the "group" that its YAML files configures for.

For example, if we needed to change what data was downloaded (e.g., how many snapshots, how large of a cube), then we would create a group/subfolder called `download/`.
- Within the subfolder, YAML keys must exactly match the dataclasses defined in `config.py`.

The root `config.yaml` declares which variation of YAML config to use for each group, and can also declare any top-level config values which don't belong in any particular group.

### Structured Configs (Type Safety)
Dataclasses in `config.py` define the expected schema for each config group. A top-level `SuperResolutionConfig` dataclass ties them all together:

```python
@dataclass
class SuperResolutionConfig:
    download: DownloadConfig = MISSING
    preprocess: PreprocessConfig = MISSING
    train: TrainConfig = MISSING
    raw_data_dir: Optional[str] = None
    processed_data_dir: Optional[str] = None
```

To get type checking and autocomplete in your IDE, register the schema with Hydra's `ConfigStore` at the top of each entry-point script:

```python
from hydra.core.config_store import ConfigStore
from .config import SuperResolutionConfig

cs = ConfigStore.instance()
cs.store(name="config", node=SuperResolutionConfig)
```

Then annotate the entry-point function with `SuperResolutionConfig` instead of `DictConfig`:

```python
@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_eval(cfg: SuperResolutionConfig):
    cfg.train.epochs  # Pylance knows this is int
```

### Running Code
Add a `@hydra.main` decorator on top of the entrypoint function:
- `version_base=None` uses latest version (avoids deprecation warnings)
- `config_path="configs"` specifies relative path from the Python script to the `configs/` folder
- `config_name="config"` points to the primary config file within that folder

```python
@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_eval(cfg: SuperResolutionConfig):
    ...
```

Config values can be overridden from the command line:
```
python -m superresolution.train train.epochs=500 train.batch_size=4
```

**Working Dir**: When Hydra runs a script, it automatically cd's into a timestamped output directory (e.g., `outputs/2026-03-01/14-30-00/`) before the code executes so that output logs are stored there. Use `Path(__file__).resolve()` instead of relative paths to reference project files.
