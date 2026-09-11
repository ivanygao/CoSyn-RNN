# Discrete Synaptic States and Context-Modulated Readouts Support Continual Learning

This repository contains the code for the experiments and analyses
presented in:

> **Discrete Synaptic States and Context-Modulated Readouts Support
> Continual Learning**\
> [Paper link]()

## Overview

This repository implements the continual learning framework proposed in
the paper, including:

-   CoSyn-RNN models with discrete synaptic states and context-modulated
    readouts
-   training and evaluation pipelines
-   dataset generation utilities
-   experiment configurations and analysis scripts

The codebase is organized to reproduce the experiments reported in the
paper.

## Installation

The code was developed and tested on Linux with NVIDIA CUDA 13 support.

Create the Conda environment from the repository root:

``` bash
conda env create -f packages/nntp/environments/linux-cuda13.yml
conda activate nntp-cuda13
```

Install the `nntp` package:

``` bash
pip install -e ./packages/nntp
```

Weights & Biases logging is optional. To enable experiment tracking:

``` bash
wandb login
```

Logging can be disabled through the experiment configuration files.

## Dataset Generation

Generate the datasets required for the experiments:

``` bash
./scripts/data/generate_data.sh
```

This script generates the training, validation, and test datasets
according to the configured tasks and random seeds.

## Running Experiments

A single experiment can be launched with:

``` bash
nntp \
  --entrypoint ./src/entrypoint.py \
  run \
  --config ./scripts/example.yaml \
  --debug
```

Remove `--debug` for a standard experiment run.

Experiment configurations are stored in:

``` text
scripts/
```

Each configuration specifies the task setup, model parameters,
optimization settings, and evaluation options.

## Reproducing Paper Results

The experiments reported in the paper can be reproduced by running the
corresponding configuration files.

Example:

``` bash
nntp \
  --entrypoint ./src/entrypoint.py \
  run \
  --config ./scripts/<experiment_config>.yaml
```

The configuration files corresponding to individual figures and
experiments are provided in the `scripts/` directory.

## Outputs

Experiment outputs include:

``` text
runtime/
├── data/           # Training datas
├── logs/           # Training logs
└── pipeline/       # Evaluation results and checkpoints
```

When enabled, additional experiment tracking information is stored
through Weights & Biases.

## Repository Structure

``` text
.
├── packages/nntp/          # Training pipeline package
│   └── environments/       # Conda environment definitions
├── src/                    # CoSyn-RNN models and experiment entry point
├── scripts/                # Dataset and Experiment configurations
├── figures/                # Paper figures
└── runtime/                # Generated experiment outputs
```

## Citation

Citation information will be added after publication.
