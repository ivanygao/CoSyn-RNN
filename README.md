# Discrete Synaptic States and Context-Modulated Readouts Support Continual Learning

This repository contains the code used for the experiments and analyses presented in:

> **Discrete Synaptic States and Context-Modulated Readouts Support Continual Learning**
> [Paper link]()

## Installation

The code was developed and tested on Linux with CUDA 13.

Create the Conda environment from the repository root:

```bash
conda env create -f packages/nntp/environments/linux-cuda13.yml
conda activate nntp-cuda13
```

Install the `nntp` package:

```bash
pip install -e ./packages/nntp
```

Weights & Biases logging is optional. To enable it, log in with:

```bash
wandb login
```

Logging can be disabled in the experiment configuration.

## Usage

### Generate datasets

```bash
./scripts/data/generate_data.sh
```

The script generates the training, validation, and test datasets required by the configured tasks and random seeds.

### Run an experiment

```bash
nntp \
  --entrypoint ./src/entrypoint.py \
  run \
  --config ./documents/example.yaml \
  --debug
```

Remove `--debug` for a standard experiment run.

## Repository Structure

```text
.
├── packages/nntp/      # Training pipeline package
├── figures/            # Paper figures
├── src/                # CoSyn-RNN models and entry point
└── scripts/            # Experiment scripts
```

## Citation

If you use this code in your research, please cite:

```bibtex
XXX
```
