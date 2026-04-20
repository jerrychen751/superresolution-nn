# Superresolution — turbulence super-resolution pipeline

Five neural architectures (three 3D CNNs + a Fourier Neural Operator + a Graph Neural Network) learning to reconstruct fine-resolution DNS turbulence from coarse observations, trained against the JHU Turbulence Database.

For the residual architectures, the model learns a correction term:

```
u_corrected = u_coarse_upsampled + model(u_coarse_upsampled)
```

`upsample_cnn` learns a direct coarse → fine mapping instead. `closure_cnn` predicts a subgrid-scale closure at coarse resolution.

---

## New-member checklist

Follow these in order. Each step links to the detailed section below.

**One-time setup**

1. [Prerequisites](#prerequisites) — GT PACE account, JHTDB token, SSH alias
2. [PACE HPC setup](#pace-hpc-setup-one-time-per-user) — conda env on scratch, clone repo, update hardcoded paths
3. [Local setup](#local-setup-dev--inference) — only needed if you want to run inference or debug locally

**First training run**

4. [Data acquisition](#first-time-data-acquisition) — one `download` + one `preprocess` per model variant
5. [Training](#training) — `sbatch --export=MODEL=<name> hpc_training.sh`
6. [Monitoring](#monitoring) — `squeue`, tail the Slurm stdout, tail the per-epoch CSV

**After training**

7. [Inference](#inference) — predictions on the held-out test split
8. Copy `weights.pth` back to your local machine for visualization / analysis

**Extending the pipeline**

9. [Adding a new model variant](#adding-a-new-model-variant) — three-file pattern (model, config, dispatch branch)

---

## Prerequisites

- **Georgia Tech PACE ICE account** — https://docs.pace.gatech.edu/
- **JHTDB token** — register at http://turbulence.pha.jhu.edu/authtoken.aspx, then set it in `configs/default.yaml` under `download.jhtdb_token`
- **GT VPN** when SSHing into PACE from off-campus
- **Python 3.10+** locally, with conda or mamba

Add a PACE SSH alias to `~/.ssh/config` so the commands below work verbatim:

```
Host pace
  HostName login-ice.pace.gatech.edu
  User <your-gt-username>
```

## Local setup (dev / inference)

Local is for running inference against downloaded weights or short debug training runs. Full training happens on HPC.

```bash
git clone <repo-url> pi-cnn
cd pi-cnn
conda create -n pi-cnn python=3.10
conda activate pi-cnn
pip install torch numpy scipy matplotlib pandas hydra-core omegaconf givernylocal torch_geometric
```

Commands are run from the repo root. `env/local.yaml` sets `storage_root` to `${hydra:runtime.cwd}/superresolution`, so artifacts (data, weights, outputs, logs) land inside `superresolution/` — the pipeline's effective project root.

## PACE HPC setup (one-time, per user)

### 1. Create a scratch directory and install conda there

Home-filesystem quota is too small for a full conda env, so we install it on scratch and point Slurm jobs at that prefix.

```bash
ssh pace

# Artifacts root — $storage_root in env/hpc.yaml
mkdir -p /storage/ice1/3/9/<YOURUSER>/superresolution

# Conda env, also on scratch because envs are multi-GB
mkdir -p /storage/ice1/3/9/<YOURUSER>/conda/envs
conda create --prefix /storage/ice1/3/9/<YOURUSER>/conda/envs/ai python=3.10
conda activate /storage/ice1/3/9/<YOURUSER>/conda/envs/ai
pip install torch numpy scipy hydra-core omegaconf givernylocal torch_geometric
```

### 2. Clone the repo into $HOME

```bash
mkdir -p $HOME/projects
cd $HOME/projects
git clone <repo-url> pi-cnn
```

The code lives under home, the data and artifacts live on scratch — keep this split in mind when you're running commands.

### 3. Update HPC-specific paths

PACE doesn't export `$SCRATCH` into Slurm job environments, so we hardcode the scratch path in two files:

- `superresolution/configs/env/hpc.yaml` — set `storage_root: /storage/ice1/3/9/<YOURUSER>/superresolution`
- `superresolution/hpc_training.sh` — set `CONDA_ENV=/storage/ice1/3/9/<YOURUSER>/conda/envs/ai`

### 4. (Optional) Symlink scratch artifacts into the code dir

The config uses absolute paths, so nothing breaks without symlinks — but they make it vastly easier to `ls` / `grep` results from inside the code checkout:

```bash
cd $HOME/projects/pi-cnn
ln -s /storage/ice1/3/9/<YOURUSER>/superresolution/data        data
ln -s /storage/ice1/3/9/<YOURUSER>/superresolution/weights     weights
ln -s /storage/ice1/3/9/<YOURUSER>/superresolution/checkpoints checkpoints
ln -s /storage/ice1/3/9/<YOURUSER>/superresolution/outputs     outputs
ln -s /storage/ice1/3/9/<YOURUSER>/superresolution/logs        logs-csv
```

The `logs-csv` alias avoids colliding with the existing `superresolution/logs/` directory, which captures Slurm stdout (separate from the per-epoch CSVs that live at `$storage_root/logs/`).

## First-time data acquisition

Run from PACE, inside the active conda env:

```bash
cd $HOME/projects/pi-cnn

# Pull raw DNS cubes from JHTDB into $storage_root/data/raw/
python -m superresolution.download env=hpc

# Preprocess, splitting into train/val/test subdirs per config ratios
for MODEL in cnn upsample_cnn closure_cnn fno gnn; do
  python -m superresolution.preprocess --config-name=$MODEL env=hpc
done
```

Download is slow (network-bound on JHTDB) but only happens once. Preprocess is idempotent — if all (input, target) pairs for a given split already exist, it skips immediately.

## Training

Submit a training job via Slurm, passing the model variant through the `MODEL` env var:

```bash
cd $HOME/projects/pi-cnn/superresolution
sbatch --export=MODEL=upsample_cnn hpc_training.sh
sbatch --export=MODEL=fno          hpc_training.sh
# ...
```

### Monitoring

```bash
squeue --me                                                   # queue state
scontrol show job <jobid>                                     # wall time, node, stdout path
tail -f logs/<jobid>.out                                      # Slurm stdout stream
tail -f /storage/ice1/3/9/<YOURUSER>/superresolution/logs/<model>/train_*.csv   # per-epoch CSV
```

The best-val-loss `weights.pth` is written live during training (overwritten whenever val loss improves) — you can pull a usable model before the job finishes if needed.

## Inference

Default run — reads from the held-out `test/` split and writes one `prediction_t{XXXX}.npy` per input into `outputs_dir`:

```bash
python -m superresolution.inference --config-name=upsample_cnn env=hpc
```

Override the inference source to evaluate on a completely separate preprocessed directory:

```bash
python -m superresolution.inference --config-name=upsample_cnn env=hpc \
  inference_data_dir=/path/to/external_test
```

## Adding a new model variant

1. Create `superresolution/models/<name>.py` containing three things (pattern-match against `cnn.py`):
   - a `<Name>Dataset(Dataset)` class that loads `(input, target)` pairs and converts to whatever tensor/graph format the model expects
   - a `make_training_pair(dns_velocity, sigma, ds_step, **kwargs)` function that `preprocess.py` imports dynamically
   - the `nn.Module` itself
2. Create `superresolution/configs/<name>.yaml`:

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
3. Add a branch to `_get_data_classes` in `train.py` mapping `<name>` to your Dataset (and to a non-default DataLoader if needed — the GNN branch shows this pattern).

---

## Reference: project layout

`storage_root` (set by `configs/env/local.yaml` or `configs/env/hpc.yaml`) is the single knob that picks where data and artifacts live. Every other path interpolates off it in `configs/default.yaml`.

### HPC layout (PACE ICE, where the real data/compute lives)

The **code** (git-cloned repo) and the **artifacts** (data, weights, logs) are on different filesystems:

```
~/projects/pi-cnn/                              CODE — home filesystem
├── superresolution/
│   ├── *.py                                    pipeline scripts
│   ├── configs/                                Hydra YAMLs
│   ├── models/                                 five model files
│   ├── logs/                                   Slurm stdout (e.g., 5016418.out)
│   └── hpc_training.sh                         sbatch entry point
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

`superresolution/` is the effective project root. `env/local.yaml` sets `storage_root: ${hydra:runtime.cwd}/superresolution`, so running `python -m superresolution.*` from the repo root places every artifact inside `superresolution/`:

```
/Users/jerry/coding/pi-cnn/                     repo root (contains team sandboxes + main pipeline)
├── superresolution/                            ← storage_root on local
│   ├── *.py, configs/, models/                 code
│   ├── data/                                   raw_data_dir + processed_data_dir (if populated)
│   │   ├── raw/
│   │   └── processed/{model}/{train,val,test}/
│   ├── weights/                                weights_dir — copied from HPC after training
│   │   └── {model}/weights.pth
│   ├── outputs/                                outputs_dir — local inference predictions
│   │   └── {model}/prediction_t{XXXX}.npy
│   ├── checkpoints/                            checkpoints_dir — if you ran training locally
│   │   └── {model}/checkpoint_epoch_{N}.pth
│   └── logs/                                   logs_dir — per-epoch CSVs (and Slurm stdout on HPC)
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

