# Run the super-resolution CNN on PACE

Use this guide if you want to train a CNN without learning the repository setup first.

## What the notebook does

The notebook takes real high-resolution JHTDB velocity cubes, creates an artificial coarse version, and trains a CNN to reconstruct the missing velocity information. Everyone uses the same 490 training, 105 validation, and 105 test cubes so model results can be compared fairly.

## 1. Make the ZIP on your laptop

Compress the whole `superresolution-nn` repository folder so the ZIP contains one top-level folder with the same name. Name the archive:

```text
superresolution-nn.zip
```

Downloaded and processed JHTDB data are not supposed to be inside this ZIP.

## 2. Start JupyterLab on PACE

Connect to the GT VPN, open PACE Open OnDemand, and launch **Jupyter** with:

| Setting | Recommended value |
|---|---|
| Python Environment | Anaconda3 2023.03 |
| Jupyter Interface | JupyterLab |
| Quality of Service | One available to your account, such as `coe-ice` |
| Node Type | H200 HGX for fastest training; A40 is sufficient |
| Nodes | 1 |
| Cores Per Node | 8 |
| GPUs Per Node | 1 |
| Memory Per Core | 8 GB |
| Hours | 6 |

The notebook uses one GPU. Requesting additional GPUs does not make it faster.

## 3. Upload and unzip the repository

In the PACE file browser, open your `scratch` folder and upload the ZIP.

Open a terminal from the running JupyterLab session and paste:

```bash
cd ~/scratch
unzip -o superresolution-nn.zip
cd superresolution-nn
ls README.md pyproject.toml scripts superresolution_experiments
```

The final command should list all four names without an error.

## 4. Prepare everything

Paste one command:

```bash
bash scripts/pace_prepare_notebook.sh
```

It automatically:

1. Creates the Python environment and Jupyter kernel.
2. Downloads 700 JHTDB cubes.
3. Retries temporary JHTDB errors without deleting completed downloads.
4. Preprocesses the cubes into CNN input/answer pairs.
5. Verifies the 490/105/105 train/validation/test split.

This may take hours the first time. If the session ends, start another PACE session, return to the extracted repository folder and run the same command again. Completed work is reused.

If you already prepared the data from an older copy of the repository, it remains under:

```text
~/scratch/superresolution/
```

The command detects and reuses it. Replacing the repository ZIP does not require downloading the data again.

Preparation is complete when the terminal prints:

```text
train: 490 input/target pairs
val: 105 input/target pairs
test: 105 input/target pairs
PACE preparation is complete.
```

## 5. Run the notebook

In JupyterLab:

1. Refresh the page.
2. Open `superresolution_experiments/cnn_training_template.ipynb`.
3. Choose **File > Save Notebook As** and put your name in the filename.
4. Edit the sections marked `YOUR CODE HERE`.
5. Give `EXPERIMENT_NAME` a unique value.
6. Select **Kernel > Change Kernel > Python (superresolution-nn)**.
7. Choose **Kernel > Restart Kernel**.
8. Choose **Run > Run All Cells**.

Tony can open `cnn_training_tony.ipynb` directly.

The first cell must say:

```text
Device: cuda
```

The data cell must say:

```text
Full team split verified: {'train': 490, 'val': 105, 'test': 105}
```

If it says `Device: cpu`, stop and select the correct kernel.

## 6. Find the results

Results are saved outside the uploaded repository at:

```text
~/scratch/superresolution/notebook_runs/<EXPERIMENT_NAME>/
```

The important files are:

- `best_weights.pth`: weights from the best validation epoch
- `history.json`: training and validation loss
- `test_metrics.json`: baseline and CNN MSE/MAE
- `experiment.json`: settings used for the run

Closing your browser, sleeping your laptop, or disconnecting the VPN does not stop training. The PACE session continues until training finishes, an error occurs, or its requested wall time expires. Do not delete the session from **My Interactive Sessions** while training.

## Should processed data be committed to Git?

No. One 128 x 128 x 128 x 3 float32 cube is about 24 MiB. The 700 raw cubes are about 16.4 GiB, and the input/target pairs are about 32.8 GiB. A complete copy is roughly 49 GiB before overhead.

That would make cloning and updating the repository impractical. GitHub is meant for the code and notebooks.

A better team setup is:

1. Generate the processed CNN dataset once.
2. Put it in PACE project or group storage that every team member can read.
3. Keep each member's model results in their own scratch directory.

Until shared storage is available, each member can run `pace_prepare_notebook.sh` once. Do not point teammates at another person's private scratch path; PACE permissions may block it.

## Common problems

- **JHTDB HTTP 503:** leave the preparation command running. It waits and retries.
- **Session ended:** launch another session and rerun the same preparation command.
- **No data pairs:** preparation has not reached the verified 490/105/105 result.
- **Kernel missing:** refresh JupyterLab after preparation finishes.
- **CUDA unavailable:** verify that the PACE session requested one NVIDIA GPU and select `Python (superresolution-nn)`.
