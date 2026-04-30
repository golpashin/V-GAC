import math
import os
import socket
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.parametrizations import weight_norm


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


########## Configuration and parameters ##########
# Notes:
# Problem formulation: controlled diffusion and cost, eqs (2.1) and (2.2)
# Methodology: critic, actor, and prox objectives in the training section
@dataclass
class Config:
    # State and action dimensions
    state_dim: int = 3
    action_dim: int = 3

    # Default problem data. These are specific to the Euler example
    dt: float = 0.001
    beta: float = 0.8
    domain_radius: float = 5.0
    target_radius: float = 0.005
    sde_noise_scale: float = 0.05
    init_radius_min: float = 0.005
    init_radius_max: float = 5.0
    max_episode_steps: int = 1000000
    terminal_exit_penalty_coef: float = 50.0

    # PPO and GAE
    gae_lambda: float = 0.93440341
    ppo_clip_eps: float = 0.05421939
    entropy_coef: float = 5.43153828e-06
    value_coef: float = 0.5

    # Optimization params
    lr_actor: float = 1.89428543e-07
    lr_critic: float = 0.000143209
    lr_prox: float = 0.00048399
    weight_decay: float = 1e-4
    max_grad_norm: float = 10.0

    # Network widths
    hidden_dims: Tuple[int, ...] = (64, 64)

    # PPO minibatching
    ppo_epochs: int = 6
    minibatch_size: int = 64

    # SPD bank in the viscosity test family, paper eq (3.3)
    alpha_min: float = 1e-2
    alpha_max: float = 1e2
    num_M_mats: int = 4

    # Loss weights in the Methodology section
    lambda_visc: float = 0.01164584
    lambda_jet: float = 6.48143900e-06
    lambda_bdy: float = 4.7971259
    lambda_env: float = 3.76658843e-05
    lambda_adv: float = 5.06402835e-06
    lambda_prox_opt: float = 0.02608128
    prox_adv_steps: int = 2

    # Anchor mixing law nu_X = (1-rho) nu_on + rho nu_cover
    rho_cover: float = 0.83593892

    # Squashed Gaussian policy parameterization
    action_limit: float = 15.0
    log_std_min: float = -5.0
    log_std_max: float = -1.5
    tanh_eps: float = 1e-6
    policy_mu_clip: float = 20.0

    # Training loop
    total_iterations: int = 400
    seed: int = 0
    num_rollout_workers: int = 16
    steps_per_worker: int = 128
    display_every_iters: int = 1