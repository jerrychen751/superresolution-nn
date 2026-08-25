This project uses [Hydra](https://hydra.cc/) (from Meta) to manage experiment configuration. It composes nested YAML files into one runtime config object and lets you override individual fields from the CLI — handy for quick experimentation.

Both are pinned in `pyproject.toml` and installed by `setup_env.sh`. To add them to an environment by hand:
```
uv add hydra-core==1.3.2 omegaconf==2.3.0
```

### Layout

```
configs/
  default.yaml       # shared baseline for every experiment
  cnn.yaml           # each model file extends default and sets its model block
  closure_cnn.yaml
  upsample_cnn.yaml
  fno.yaml
  gnn.yaml
  env/
    local.yaml       # storage_root for local development
    hpc.yaml         # storage_root for PACE / HPC (hardcoded scratch path)
```

`default.yaml` holds the download, preprocess, and train blocks that every experiment uses. It also defines the artifact paths (`raw_data_dir`, `processed_data_dir`, `checkpoints_dir`, etc.) as interpolations off a `storage_root` variable — the env config group supplies the concrete value.

The `env/` subdirectory is a Hydra [config group](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/). Each file inside is an alternative; the top-level config names which one to compose via its `defaults:` list.

### Top-level config structure

```yaml
# cnn.yaml
defaults:
  - default
  - env: local
  - _self_

model:
  name: cnn
  params:
    _target_: superresolution.models.cnn.SuperResolutionCNN
    hidden_channels: 32
    num_blocks: 4
```

The `_target_` field tells `hydra.utils.instantiate()` which class to construct; the rest of `params` becomes constructor kwargs. `model.name` is a plain string used by `preprocess.py` to pick the right `make_training_pair` function.

### How storage_root works

Each env file sets `storage_root` to an absolute path. `default.yaml` interpolates all artifact paths off that root:

```yaml
# env/local.yaml
# @package _global_
storage_root: ${hydra:runtime.cwd}/superresolution   # artifacts live under superresolution/ — the effective project root
```

```yaml
# env/hpc.yaml
# @package _global_
storage_root: /storage/ice1/3/9/<YOURUSER>/superresolution   # hardcoded; $SCRATCH isn't exported into Slurm jobs
```

```yaml
# default.yaml (excerpt)
raw_data_dir: ${storage_root}/data/raw
processed_data_dir: ${storage_root}/data/processed/${model.name}
inference_data_dir: ${processed_data_dir}/test
checkpoints_dir: ${storage_root}/checkpoints/${model.name}
weights_dir: ${storage_root}/weights/${model.name}
outputs_dir: ${storage_root}/outputs/${model.name}
logs_dir: ${storage_root}/logs/${model.name}
```

The `# @package _global_` directive at the top of each env file tells Hydra to merge its contents at the root of the composed config rather than nesting them under an `env` key.

### Running

Each entry-point script uses `@hydra.main` with `config_name="cnn"` as the default root, so running with no arguments uses `cnn.yaml`. Pass `--config-name=<model>` to select a different experiment:

```bash
python -m superresolution.train                           # cnn (default)
python -m superresolution.train --config-name=gnn         # gnn
python -m superresolution.preprocess --config-name=fno    # fno
```

Override individual fields with standard Hydra syntax:

```bash
python -m superresolution.train --config-name=cnn train.epochs=500 train.batch_size=4
python -m superresolution.download raw_data_dir=/path/to/raw
```

Switch environments by overriding the env group:

```bash
python -m superresolution.train --config-name=cnn env=hpc
```

On HPC, `hpc_training.sh` passes `env=hpc` to every step so all artifacts land on scratch automatically.

### Inspecting a composed config

To see exactly what Hydra will pass to your code — with all interpolations resolved — use `--cfg job --resolve`:

```bash
python -m superresolution.train --config-name=cnn --cfg job --resolve
```

This prints the final config and exits without running anything. Handy for sanity-checking overrides before you commit to a real run.

### Hydra changes your working directory

At startup, Hydra `cd`s into a timestamped `outputs/YYYY-MM-DD/HH-MM-SS/` so each run's auto-generated logs stay isolated. Inside any `@hydra.main`-decorated function, always resolve file paths via the config (e.g. `Path(cfg.checkpoints_dir)`) — relative paths will resolve against the timestamped dir, not where you ran the command from.
