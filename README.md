# OVG

Official PyTorch implementation of Orthogonal Variance Guidance from ***Spectral Collapse in Diffusion Inversion***. Uses Hydra for configuration, Accelerate for training/benchmark orchestration, and logs metrics/visuals via wandb.

## Features

- Training and evaluation workflows with Hydra configs
- Benchmark runner with image quality, distribution, and spectral analyses
- Support for BBBC021 and Edges2Shoes datasets
- Optional HDF5 export of intermediate latents and predicted noise

## Installation

This repository is a Python package (see `pyproject.toml`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[notebook]"
pip install -e ".[hpc]"
```

## Quickstart

Train with Hydra configs:

```bash
python train.py --config-path configs/bbbc021 --config-name train
```

Run a benchmark/evaluation:

```bash
python test.py --config-path configs/bbbc021 --config-name test
```

## Configuration

Configs live under `configs/` and are composed with Hydra. Each dataset has `train.yaml`, `test.yaml`, and transform definitions in `transforms/`.

Useful Hydra tips:

```bash
# Override a value
python train.py --config-path configs/bbbc021 --config-name train data.batch_size=4

# Print the final composed config
python train.py --config-path configs/bbbc021 --config-name train --cfg job
```

## Project Structure

- `train.py`: training entry point
- `test.py`: benchmark entry point
- `benchmarker.py`: benchmark runner and metrics/plots
- `configs/`: Hydra configs
- `data/`: datasets and transforms
- `models/`: model components
- `schedulers/`: scheduler adapters
- `training/`: training strategies
- `utils/`: logging, checkpoints, metrics, helpers

## Datasets

Supported datasets are:

- `bbbc021`
- `edges2shoes`

Dataset paths and modes are configured in the dataset-specific configs under `configs/`.
