# pi-cnn

`pi-cnn` is a PyTorch project for turbulence super-resolution. The primary
pipeline learns to reconstruct fine-resolution direct numerical simulation
(DNS) velocity fields from coarse observations obtained from the JHU Turbulence
Database (JHTDB).

The maintained training pipeline lives in [`superresolution/`](superresolution/).
It supports 3D CNN, upsampling CNN, closure CNN, Fourier neural operator (FNO),
and graph neural network (GNN) variants. The other top-level team folders hold
useful prototypes, notebooks, and prior investigations; they are not the
canonical training path.

## Start here

New contributors should begin with the package README:

```bash
cd superresolution-nn
bash setup_env.sh
source .venv/bin/activate

# Inspect the resolved configuration without starting a run.
python -m superresolution.train --config-name=cnn --cfg job --resolve
```

The project requires Python 3.11 and uses `uv` to install locked dependencies.
The setup script installs `uv` when needed. On macOS and Windows, PyTorch is
installed with CPU support by default; full training is intended for the PACE
GPU cluster.

Before downloading data, obtain a JHTDB access token and set
`download.jhtdb_token` locally in `superresolution/configs/default.yaml`. Do
not commit a personal token. Prefer a local, uncommitted configuration override
when working with shared branches.

## Main workflow

Run commands from the repository root. Hydra selects a model configuration with
`--config-name=<model>` and accepts overrides as `key=value` arguments.

```bash
# 1. Download raw velocity cubes from JHTDB (one-time per dataset).
python -m superresolution.download env=hpc

# 2. Build train/validation/test data for a model.
python -m superresolution.preprocess --config-name=cnn env=hpc

# 3. Train with the prepared data.
python -m superresolution.train --config-name=cnn env=hpc

# 4. Generate predictions from the best saved weights.
python -m superresolution.inference --config-name=cnn env=hpc
```

For production-size jobs on PACE, use the Slurm wrappers instead:

```bash
cd superresolution
sbatch hpc_preprocess.sh
sbatch --export=MODEL=cnn hpc_training.sh
```

See [`superresolution/README.md`](superresolution/README.md) for PACE setup,
storage paths, monitoring jobs, inference details, and the complete data layout.

## Repository map

| Path | Purpose |
| --- | --- |
| [`superresolution/`](superresolution/) | Maintained download, preprocessing, training, and inference pipeline. |
| [`superresolution/configs/`](superresolution/configs/) | Hydra model and environment configurations. |
| [`superresolution/models/`](superresolution/models/) | Model architectures plus their dataset and input/output adapters. |
| [`superresolution/docs/`](superresolution/docs/) | Design notes for models and distributed training. |
| [`superresolution_experiments/`](superresolution_experiments/) | Image-filtering, JHTDB, and GNN experiments retained for reference. |
| [`jerry/`](jerry/) | Individual prototypes and learning notebooks. |
| [`yash/`](yash/) | Autoencoder/GAN exploration and archived comparison work. |
| [`data/`](data/) | Local data cache; generated data is ignored by Git. |

## Configuration and artifacts

`superresolution/configs/default.yaml` defines shared download, preprocessing,
and training settings. Individual files such as `cnn.yaml`, `fno.yaml`, and
`gnn.yaml` choose the architecture and its parameters. The `env` config group
selects where generated artifacts are stored:

- `env=local` writes artifacts beneath `superresolution/` for local debugging
  and inference.
- `env=hpc` points artifacts at PACE scratch storage for large data and GPU jobs.

Generated raw data, processed data, checkpoints, weights, predictions, logs,
and local JHTDB output are intentionally ignored by Git. Keep source code,
configs, and lightweight documentation under version control.

## Contributing

Keep new work easy to reproduce and easy to find:

1. Make pipeline changes in `superresolution/`, not in an experiment folder.
2. Use a model-specific Hydra config for tunable parameters rather than
   hardcoding values in Python.
3. Keep paths derived from the config values such as `cfg.processed_data_dir`;
   Hydra changes the working directory for each run.
4. Record the model, config overrides, dataset source, and result summary when
   sharing an experiment or opening a pull request.
5. Add focused tests when changing reusable data transformations, model shapes,
   or training behavior. There is currently no committed automated test suite,
   so also run a small local smoke test or validate the composed Hydra config.

### Adding a model

A model variant needs four pieces:

1. A module in `superresolution/models/` containing the `nn.Module`, dataset,
   `make_training_pair`, and `build_inference_fn` implementations.
2. A Hydra config in `superresolution/configs/` that defines `model.name` and
   `model.params`.
3. Dispatch entries in `preprocess.py`, `train.py`, and `inference.py`.
4. A small documented command that preprocesses, trains, and runs inference for
   the new variant.

The detailed model-extension guide is in
[`superresolution/README.md`](superresolution/README.md).

## Useful references

- [`superresolution/README.md`](superresolution/README.md): operational guide
  for local development and PACE.
- [`superresolution/configs/README.md`](superresolution/configs/README.md):
  Hydra configuration layout and overrides.
- [`superresolution/docs/model.md`](superresolution/docs/model.md): model
  integration notes.
- [`superresolution/docs/ddp.md`](superresolution/docs/ddp.md): distributed
  training background.
