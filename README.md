# Bridging Complexity in Neural Dynamics

This repository contains resources and instructions to be used for the tutorial of the CNSP 2025 workshop presented by H. Weissbart (on the 2nd of Spectember at 6.30pm IST).

## Objectives

This tutorial presents advanced methodologies to extend **Temporal Response Function (TRF)** analysis beyond
its usual linear framework, deepening our understanding of complex neural interactions. Traditionally,
TRF derives “evoked potential”-like responses from continuous recordings with arbitrary stimulus signals,
accommodating diverse event and predictor structures. We briefly review fundamental usage, including model
equation solutions and regularisation strategies—such as incorporating prior spectral information in singular
value decomposition-based TRF solutions.
Building on this, we propose using non-linear mappings to synthesise more complex signals, empowering linear
convolutional models to extract precise temporal responses to sophisticated neural features, such as phase
clustering and phase-amplitude coupling. We also introduce a technique that modulates samples
using ongoing neural phase or other stimulus-bound data, weighting samples entering the TRF decomposition,
and enabling hierarchical representations of stimulus dependencies.

## Content

There are both data and notebooks accessible here, either directly or via simple terminal commands/programs. The repository is organised as follows:

- `data`: data used in the tutorial, further instructions are detailed in `data/readme.md`
- `notebooks`: the heart of the tutorial
- `utils`: a collection of modules and scripts that are sometimes used in the tutorial notebooks

## Setup

We will be using python >3.10*, to keep a modern version.
You can start with a fresh environment if you like, using either `requirements.txt` via `pip` or the `environment.yaml` via `conda` or `mamba`:

- Using `pip`:
  ```bash
  # If installing in a new environment:
  mamba create -n myenv python>=3.10 pip # or conda/venv/etc.
  mamba activate myenv # or conda/venv etc.
  # You can go straight to `pip install` if installing in an existing env:
  pip install -r requirements.txt
  ```

- Using `mamba` (or `conda`, simply replace `mamba` by `conda` in the command below):
  ```bash
  mamba create -n myenv -f environment.yaml
  ```

