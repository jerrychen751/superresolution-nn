# pi-cnn

This repository contains the team's computational fluid dynamics and machine-learning experiments. The maintained Python package in `src/superresolution/` downloads JHTDB velocity fields, prepares coarse-to-fine training pairs, and trains super-resolution models.

## Start here

| Goal | Guide |
|---|---|
| Run or modify the CNN notebook on Georgia Tech PACE | [Super-resolution notebook guide](superresolution_experiments/README.md) |
| Understand how preprocessing creates the input and answer | [CNN preprocessing](docs/CNN_PREPROCESSING.md) |
| Work on the maintained training pipeline | [Pipeline reference](docs/PIPELINE_REFERENCE.md) |
| Add a new model to the shared pipeline | [Model guide](docs/model.md) |

## Repository map

| Path | Contents |
|---|---|
| `src/superresolution/` | Maintained download, preprocessing, training, inference, and model code |
| `superresolution_experiments/` | Beginner notebooks and exploratory super-resolution work |
| `scripts/` | PACE setup, preprocessing, and training scripts |
| `docs/` | Technical explanations and reference material |
| `jerry/`, `yash/` | Individual research and learning work |

## Local development

Install the project and notebook dependencies from the repository root:

```bash
uv sync --group extras
```

Run Python entry points through the project environment:

```bash
uv run python -m superresolution.download
uv run python -m superresolution.preprocess --config-name=cnn
```

The full real dataset belongs on PACE scratch or shared project storage. Do not commit downloaded cubes, processed arrays, model weights, or access tokens to Git.
