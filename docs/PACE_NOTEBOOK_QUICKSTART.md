# PACE notebook quickstart

This guide is for a new team member who has a PACE account and a ZIP of this
repository. It prepares that member's own environment and data. It does not
read or write another member's PACE storage.

## 1. Start a PACE Jupyter session

Open PACE Open OnDemand and launch **JupyterLab**. For the simplest single-session
workflow, request:

- Nodes: `1`
- Node type: `A40 NVIDIA GPU` (or another available NVIDIA GPU)
- CPU cores: `4`
- GPUs: `1`
- Memory per core: `8–16 GB` (32–64 GB total)
- Hours: the longest practical allocation; downloading, preprocessing, and 100
  training epochs can exceed a short session
- Quality of Service: one available to your PACE account

Downloading and preprocessing use the CPU. To conserve GPU time, you may run
steps 2–3 in a CPU-only Jupyter session, end it after preparation completes,
then launch a GPU session for step 4. Files and the registered kernel persist
between sessions.

Do not run the preparation commands on a PACE login node.

## 2. Upload and unzip the repository

Use the PACE Files interface to upload `superresolution-nn.zip` into your
`scratch` directory. Open a terminal in the Jupyter session and paste:

```bash
cd ~/scratch
unzip -o superresolution-nn.zip
cd superresolution-nn
```

The ZIP should contain one top-level `superresolution-nn/` directory. Confirm
that the terminal is in the right place:

```bash
pwd
ls README.md pyproject.toml scripts superresolution_experiments
```

## 3. Create the kernel, download JHTDB, and preprocess

Paste this one command:

```bash
bash scripts/pace_prepare_notebook.sh
```

The script performs the complete one-time preparation:

1. Finds your own PACE scratch directory.
2. Creates or updates `scratch/venvs/superresolution-nn` using `uv`.
3. Registers the **Python (superresolution-nn)** Jupyter kernel.
4. Downloads the same 700 JHTDB cubes used by the team.
5. Automatically restarts the downloader after temporary JHTDB errors. Saved
   cubes are skipped, so progress is preserved.
6. Creates the CNN input/target arrays.
7. Verifies the fixed `490/105/105` train/validation/test split.

For an explanation of what happens to each velocity cube, see
**[CNN preprocessing: DNS velocity to training pairs](CNN_PREPROCESSING.md)**.

The download uses the team's configured JHTDB token. Do not paste the token
into a notebook, terminal command, or chat.

If the PACE session ends or you press `Ctrl+C`, start another compute session,
return to the repository, and run the same command again:

```bash
cd ~/scratch/superresolution-nn
bash scripts/pace_prepare_notebook.sh
```

Environment setup, download, and preprocessing are resumable. Existing work is
reused rather than deleted.

Successful preparation ends with:

```text
train: 490 input/target pairs
val: 105 input/target pairs
test: 105 input/target pairs
PACE preparation is complete.
```

## 4. Copy the template and run training

In JupyterLab:

1. Refresh the browser page so the new kernel appears.
2. Open `superresolution_experiments/cnn_training_template.ipynb`.
3. Use **File → Save Notebook As** and name it after yourself, such as
   `cnn_training_alex.ipynb`.
4. Edit only the cells marked `YOUR CODE HERE`.
5. Set a unique `EXPERIMENT_NAME`.
6. Select **Kernel → Change Kernel → Python (superresolution-nn)**.
7. Restart the kernel.
8. Click **Run → Run All Cells**.

The first cell must report `Device: cuda`. The data cell must report the current
member's own scratch path and confirm:

```text
Full team split verified: {'train': 490, 'val': 105, 'test': 105}
```

Results are written to:

```text
~/scratch/superresolution/notebook_runs/<EXPERIMENT_NAME>/
```

The result directory contains the best model weights, training history, test
metrics, and experiment metadata.

## Tony's implemented notebook

Tony can open `superresolution_experiments/cnn_training_tony.ipynb` instead of
copying the template. The preparation steps and kernel selection are otherwise
the same.

## Common problems

- **HTTP 503 from JHTDB:** leave the preparation script running. It reports the
  saved cube count, waits 15 seconds, and resumes automatically.
- **`Device: cpu`:** the notebook is using the wrong kernel or the Jupyter
  session has no GPU. Select `Python (superresolution-nn)` and confirm the PACE
  session requested one NVIDIA GPU.
- **No input/target pairs:** preparation has not completed. Rerun
  `bash scripts/pace_prepare_notebook.sh` and wait for all three verified counts.
- **Session ended:** request a new compute session and rerun the same preparation
  command. Scratch files persist.
- **Future dataset:** update `DATASET_NAME` and `EXPECTED_SPLIT_SIZES` in the
  notebook, or set `EXPECTED_SPLIT_SIZES = None` for an exploratory dataset.
