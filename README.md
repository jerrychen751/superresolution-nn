# superresolution-nn

`superresolution-nn` is a PyTorch project for turbulence super-resolution. The primary
pipeline learns to reconstruct fine-resolution direct numerical simulation
(DNS) velocity fields from coarse observations obtained from the JHU Turbulence
Database (JHTDB).

The maintained training pipeline lives in [`src/superresolution/`](src/superresolution/).
It supports 3D CNN, upsampling CNN, closure CNN, Fourier neural operator (FNO),
and graph neural network (GNN) variants. The other top-level team folders hold
useful prototypes, notebooks, and prior investigations; they are not the
canonical training path.

## Repository map

| Path | Purpose |
| --- | --- |
| [`src/superresolution/`](src/superresolution/) | Maintained download, preprocessing, training, and inference pipeline. |
| [`src/superresolution/configs/`](src/superresolution/configs/) | Hydra model and environment configurations. |
| [`src/superresolution/models/`](src/superresolution/models/) | Model architectures plus their dataset and input/output adapters. |
| [`scripts/`](scripts/) | Environment setup and Slurm entry points. |
| [`docs/`](docs/) | Design notes for models and distributed training. |
| [`superresolution_experiments/`](superresolution_experiments/) | Image-filtering, JHTDB, and GNN experiments retained for reference. |
| [`jerry/`](jerry/) | Individual prototypes and learning notebooks. |
| [`yash/`](yash/) | Autoencoder/GAN exploration and archived comparison work. |
| [`data/`](data/) | Local data cache; generated data is ignored by Git. |

Five neural architectures (three 3D CNNs + a Fourier Neural Operator + a Graph Neural Network) learning to reconstruct fine-resolution DNS turbulence from coarse observations, trained against the JHU Turbulence Database.

For the residual architectures, the model learns a correction term:

```
u_corrected = u_coarse_upsampled + model(u_coarse_upsampled)
```

`upsample_cnn` learns a direct coarse → fine mapping instead. `closure_cnn` predicts a subgrid-scale closure at coarse resolution.

## If you're new

One-time: get your accounts set up, create the uv environment on PACE scratch, clone the repo, and plug your scratch path into three HPC config files. Then for each model, `download` once (ever) and `preprocess` once, then sbatch a training job. When that lands, run inference on PACE or copy `weights.pth` down and run it locally. Sections below walk through each piece.

## Prerequisites

- **Georgia Tech PACE ICE account** — https://docs.pace.gatech.edu/
- **JHTDB token** — register at http://turbulence.pha.jhu.edu/authtoken.aspx, then set it in `src/superresolution/configs/default.yaml` under `download.jhtdb_token`
- **GT VPN** when SSHing into PACE from off-campus
- **[uv](https://docs.astral.sh/uv/)** locally — `curl -LsSf https://astral.sh/uv/install.sh | sh`. It installs its own Python, so no system Python or conda is needed.

Add a PACE SSH alias to `~/.ssh/config` so the commands below work verbatim:

```
Host pace
  HostName login-ice.pace.gatech.edu
  User <your-gt-username>
```

## Local setup (dev / inference)

Local setup is only for running inference against downloaded weights or short debug runs on a tiny sample. All real training stays on HPC.

```bash
git clone <repo-url> superresolution-nn
cd superresolution-nn
bash scripts/hpc_env_setup.sh
source .venv/bin/activate
```

`scripts/hpc_env_setup.sh` reads `pyproject.toml` and `uv.lock`, so everyone gets the same versions. On Linux uv pulls the CUDA 12.6 torch wheels; on macOS and Windows it takes the CPU wheels from PyPI.

`scripts/hpc_env_setup.sh` also installs the `superresolution` package itself in editable mode, so `python -m superresolution.*` resolves from any directory. Commands are still run from the repo root, because `env/local.yaml` sets `storage_root` to `${hydra:runtime.cwd}` — artifacts (data, weights, outputs, logs) land beside `src/` at the repo root.

## PACE HPC setup (one-time, per user)

### 1. Clone the repo into $HOME

```bash
ssh pace
mkdir -p $HOME/projects
cd $HOME/projects
git clone <repo-url> superresolution-nn
```

The code lives under home, the data and artifacts live on scratch — keep this split in mind when you're running commands.

### 2. Create a scratch directory and build the environment there

Home-filesystem quota is too small for the venv, uv's wheel cache, or the interpreter uv downloads, so all three go on scratch and Slurm jobs point at that prefix.

```bash
# Artifacts root — $storage_root in env/hpc.yaml
mkdir -p /storage/ice1/3/9/<YOURUSER>/superresolution

curl -LsSf https://astral.sh/uv/install.sh | sh
cd $HOME/projects/superresolution-nn
VENV=/storage/ice1/3/9/<YOURUSER>/venvs/superresolution-nn bash scripts/hpc_env_setup.sh
```

Point `VENV` outside `$HOME` and `scripts/hpc_env_setup.sh` puts the wheel cache and the interpreter beside it automatically. No `module load anaconda3` is needed: `uv python install` fetches its own CPython.

### 3. Update HPC-specific paths

PACE doesn't export `$SCRATCH` into Slurm job environments, so we hardcode the scratch path in three files:

- `src/superresolution/configs/env/hpc.yaml` — set `storage_root: /storage/ice1/3/9/<YOURUSER>/superresolution`
- `scripts/hpc_training.sh` — set `VENV=/storage/ice1/3/9/<YOURUSER>/venvs/superresolution-nn`
- `scripts/hpc_preprocess.sh` — set the same `VENV`

### 4. (Optional) Symlink scratch artifacts into the code dir

The config uses absolute paths, so nothing breaks without symlinks — but they make it vastly easier to `ls` / `grep` results from inside the code checkout:

```bash
cd $HOME/projects/superresolution-nn
ln -s /storage/ice1/3/9/<YOURUSER>/superresolution/data        data-scratch
ln -s /storage/ice1/3/9/<YOURUSER>/superresolution/weights     weights
ln -s /storage/ice1/3/9/<YOURUSER>/superresolution/checkpoints checkpoints
ln -s /storage/ice1/3/9/<YOURUSER>/superresolution/outputs     outputs
ln -s /storage/ice1/3/9/<YOURUSER>/superresolution/logs        logs-csv
```

The `data-scratch` and `logs-csv` aliases avoid two directories the repo already tracks: `data/` (MNIST fixtures) and `logs/` (Slurm stdout). The per-epoch CSVs live separately under `$storage_root/logs/`.

## Getting the data

On PACE, with the venv activated:

```bash
cd $HOME/projects/superresolution-nn

# Pull raw DNS cubes from JHTDB into $storage_root/data/raw/
python -m superresolution.download env=hpc

# Preprocess, splitting into train/val/test subdirs per config ratios
for MODEL in cnn upsample_cnn closure_cnn fno gnn; do
  python -m superresolution.preprocess --config-name=$MODEL env=hpc
done
```

Download is slow (network-bound on JHTDB) but only happens once. Preprocess is safe to rerun — each pair already on disk is skipped, so a job killed at wall time resumes where it stopped.

The loop above runs on the login node. To run it as a batch job instead:

```bash
cd $HOME/projects/superresolution-nn
sbatch scripts/hpc_preprocess.sh
```

The `cd` matters: `#SBATCH --output=logs/%j.out` resolves against the directory you submit from, and Slurm does not create it. Either way, download must have run first, because preprocess exits non-zero when the raw directory is empty.

## Training

Submit a training job via Slurm, passing the model variant through the `MODEL` env var:

```bash
cd $HOME/projects/superresolution-nn
sbatch --export=MODEL=upsample_cnn scripts/hpc_training.sh
sbatch --export=MODEL=fno          scripts/hpc_training.sh
# ...
```

### Monitoring

```bash
squeue --me                                                   # queue state
scontrol show job <jobid>                                     # wall time, node, stdout path
tail -f logs/<jobid>.out                                      # Slurm stdout stream
tail -f /storage/ice1/3/9/<YOURUSER>/superresolution/logs/<model>/train_*.csv   # per-epoch CSV
```

`weights.pth` gets overwritten every time val loss improves, so you can pull a usable model before the job hits wall time — no need to wait.

## Inference

By default, inference reads from the held-out `test/` split and writes one `prediction_t{XXXX}.npy` per input into `outputs_dir`:

```bash
python -m superresolution.inference --config-name=upsample_cnn env=hpc
```

Override the inference source to evaluate on a completely separate preprocessed directory:

```bash
python -m superresolution.inference --config-name=upsample_cnn env=hpc \
  inference_data_dir=/path/to/external_test
```

## Adding a new model variant

Look at `src/superresolution/models/cnn.py` as the reference — everything below is the pattern it follows.

1. Create `src/superresolution/models/<name>.py` with four things in it:
   - the `nn.Module` itself
   - a `<Name>Dataset(Dataset)` class that loads `(input, target)` pairs and converts to whatever tensor or graph format the model expects
   - a `make_training_pair(dns_velocity, sigma, ds_step, **kwargs)` function — `preprocess.py` imports this dynamically
   - a `build_inference_fn(model, sample_shape, device)` factory that returns a `predict(input_array) -> prediction_array` closure — `inference.py` imports this dynamically
2. Create `src/superresolution/configs/<name>.yaml`:

   ```yaml
   defaults:
     - default
     - env: local
     - _self_

   model:
     name: <name>
     params:
       _target_: superresolution.models.<name>.<ModelClass>
       # constructor kwargs
   ```
3. Wire up the three dispatch points:
   - `_get_data_classes` in `train.py` — return your Dataset (and a non-default DataLoader if needed, as GNN does)
   - `_get_build_inference_fn` in `inference.py` — return your `build_inference_fn`
   - the import block in `preprocess.py` — import your `make_training_pair`

---

## Reference: project layout

`storage_root` (set by `src/superresolution/configs/env/local.yaml` or `src/superresolution/configs/env/hpc.yaml`) is the single knob that picks where data and artifacts live. Every other path interpolates off it in `src/superresolution/configs/default.yaml`.

### HPC layout (PACE ICE, where the real data/compute lives)

The **code** (git-cloned repo) and the **artifacts** (data, weights, logs) are on different filesystems:

```
~/projects/superresolution-nn/                  CODE — home filesystem
├── src/superresolution/
│   ├── *.py                                    pipeline scripts
│   ├── configs/                                Hydra YAMLs
│   └── models/                                 five model files
├── scripts/
│   ├── hpc_training.sh                         sbatch entry point: preprocess + train one model
│   └── hpc_preprocess.sh                       sbatch entry point: preprocess all five models
├── logs/                                       Slurm stdout (e.g., 5016418.out)
├── docs/
└── CLAUDE.md

/storage/ice1/3/9/jchen3421/superresolution/             ARTIFACTS — scratch filesystem
│                                               ↑ this is storage_root on HPC
├── data/
│   ├── raw/                                    raw_data_dir
│   │   └── velocity_t{XXXX}.npy                DNS cubes sampled from JHTDB
│   └── processed/
│       └── {model}/                            processed_data_dir — one per model
│           ├── train/                          weight updates read from here
│           │   ├── input_t{XXXX}.npy
│           │   └── target_t{XXXX}.npy
│           ├── val/                            per-epoch val loss, drives weights.pth
│           │   ├── input_t{XXXX}.npy
│           │   └── target_t{XXXX}.npy
│           └── test/                           held-out, inference_data_dir default
│               ├── input_t{XXXX}.npy
│               └── target_t{XXXX}.npy
├── checkpoints/
│   └── {model}/                                checkpoints_dir
│       └── checkpoint_epoch_{N}.pth            periodic snapshot, every N epochs
├── weights/
│   └── {model}/                                weights_dir
│       └── weights.pth                         single file, overwritten any epoch whose
│                                               val_loss beats the running best (not tied
│                                               to the checkpoint cadence)
├── logs/
│   └── {model}/                                logs_dir
│       └── train_{YYYYMMDD_HHMMSS}.csv         per-epoch train/val/lr log
└── outputs/
    └── {model}/                                outputs_dir — inference predictions
```

### Local layout (your Mac)

The repo root is `storage_root`. `env/local.yaml` sets `storage_root: ${hydra:runtime.cwd}`, so running `python -m superresolution.*` from the repo root places every artifact beside `src/`:

```
/Users/jerry/coding/superresolution-nn/         repo root — also storage_root on local
├── src/superresolution/                        code
│   └── *.py, configs/, models/
├── scripts/                                    sbatch entry points
├── docs/
├── data/                                       raw_data_dir + processed_data_dir (if populated)
│   ├── raw/
│   └── processed/{model}/{train,val,test}/
├── weights/                                    weights_dir — copied from HPC after training
│   └── {model}/weights.pth
├── outputs/                                    outputs_dir — local inference predictions
│   └── {model}/prediction_t{XXXX}.npy
├── checkpoints/                                checkpoints_dir — if you ran training locally
│   └── {model}/checkpoint_epoch_{N}.pth
├── logs/                                       logs_dir — per-epoch CSVs (and Slurm stdout on HPC)
├── jerry/, tony/, yash/                        team experiment sandboxes
└── CLAUDE.md
```

Most of these dirs aren't populated locally — full training and preprocessing happen on HPC. What you typically have locally is the `weights/` subtree (copied from HPC for inference) and whatever `outputs/` you've generated against those weights.

### Config key → actual path

| Config key (from `default.yaml`) | Interpolates to | Example on HPC |
|---|---|---|
| `storage_root` | (env-specific) | `/storage/ice1/3/9/jchen3421/superresolution` |
| `raw_data_dir` | `${storage_root}/data/raw` | `.../superresolution/data/raw/` |
| `processed_data_dir` | `${storage_root}/data/processed/${model.name}` | `.../data/processed/cnn/` |
| `inference_data_dir` | `${processed_data_dir}/test` | `.../data/processed/cnn/test/` |
| `checkpoints_dir` | `${storage_root}/checkpoints/${model.name}` | `.../checkpoints/cnn/` |
| `weights_dir` | `${storage_root}/weights/${model.name}` | `.../weights/cnn/` |
| `logs_dir` | `${storage_root}/logs/${model.name}` | `.../logs/cnn/` |
| `outputs_dir` | `${storage_root}/outputs/${model.name}` | `.../outputs/cnn/` |

### Two "outputs" directories — easy to confuse

- **`cfg.outputs_dir`** (the config key above) = where inference.py writes predictions.
- **Hydra's auto-created `outputs/`** = a timestamped `outputs/YYYY-MM-DD/HH-MM-SS/` dir that Hydra makes at startup as its cwd, for its own job logs and the resolved config snapshot. Lives next to wherever you ran the command from. Completely separate from `cfg.outputs_dir`.

The CLAUDE.md "Hydra working directory caveat" warns about the cwd-changing behavior — that's why every script resolves paths via `Path(cfg.<name>)` rather than relative paths.

### How the train / val / test split works

The split is done **at preprocess time**, not at training time. `preprocess.py` reads the raw cubes, sorts them chronologically, and writes each preprocessed pair into the `train/`, `val/`, or `test/` subdir according to the ratios in config. The filesystem itself then enforces who sees what: `train.py` only ever reads from `train/` and `val/`; `inference.py` only reads from `test/` (unless overridden).

Ratios live in the `preprocess` config block:

```yaml
preprocess:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15     # explicit, redundant but easier to read
```

These must sum to 1.0 (checked by preprocess.py). `test_ratio` isn't directly used to count files — the test split gets whatever files remain after train and val are sliced off, so rounding never drops a file. It's still listed explicitly in config so a reader can see the whole split at a glance.

**Roles of each split:**

- **train** — weights are updated from gradients on this set
- **val** — used each epoch to compute `val_loss`, which drives best-weight selection (the overwrite of `weights.pth`). The model doesn't backprop through val but is *indirectly* shaped by it via checkpoint selection.
- **test** — the held-out set. Never touched during training. `inference.py` evaluates on it by default.

**Overriding the inference source:**

```bash
# default — reads data/processed/{model}/test/
python -m superresolution.inference --config-name=upsample_cnn

# point at a completely separate preprocessed dataset (e.g. a different JHTDB download)
python -m superresolution.inference --config-name=upsample_cnn \
  inference_data_dir=/path/to/external_eval
```

`num_cubes` in `download.py` is unchanged — you still download the full set of raw cubes; the split happens entirely during preprocessing.

## Contributing

Keep new work easy to reproduce and easy to find:

1. Make pipeline changes in `src/superresolution/`, not in an experiment folder.
2. Use a model-specific Hydra config for tunable parameters rather than
   hardcoding values in Python.
3. Keep paths derived from the config values such as `cfg.processed_data_dir`;
   Hydra changes the working directory for each run.
4. Record the model, config overrides, dataset source, and result summary when
   sharing an experiment or opening a pull request.
5. Add focused tests when changing reusable data transformations, model shapes,
   or training behavior. There is currently no committed automated test suite,
   so also run a small local smoke test or validate the composed Hydra config.
