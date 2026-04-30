# V-GAC

**Viscosity-Informed Generative Actor-Critic for High-Dimensional Stochastic Optimal Control**
This repository contains the implementation of the Viscosity-Informed Generative Actor-Critic (V-GAC) method.

The included example is the stochastic Euler rigid-body stabilization control problem. The code is organized so that the problem/domain section can be replaced with other examined examples, while keeping the method core unchanged.

## Repository layout

- `vgac_core.py` — main training driver
- `config.py` — configuration and parameters
- `problem.py` — problem, domain, environment, and GPU model definitions
- `networks.py` — actor, critic, and proximal networks
- `method.py` — V-GAC method core: SPD bank, contact jets, viscosity loss, proximal loss, and jet-alignment loss
- `runtime.py` — DDP and rollout-worker infrastructure
- `train.py` — training loop
- `environment.yml` — software environment specification

## Requirements

- Linux
- NVIDIA GPU(s)
- CUDA-compatible PyTorch
- Conda, Mamba, or Micromamba

## Create the environment

Using micromamba:

```bash
micromamba create -f environment.yml -y
