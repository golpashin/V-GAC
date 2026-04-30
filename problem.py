import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import Config


########## Environment ##########
# See subsection 4.1
# To replace the environment, you may only edit this section and build_problem(cfg)
# The training loop below should not need any other changes
# Note: The hooks determine how anchors are projected, sampled, and tested on the boundary
class DomainAdapter:

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # Project a batch of states back to the closed domain
    def project(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # Sample anchor states from the covering distribution nu_cov
    def sample_cover_states(self, num: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    # Mark contact points that lie on the boundary, where the hinge is set to zero
    def boundary_contact_mask(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # Sample boundary or target points for the critic boundary penalty
    def boundary_penalty(self, critic: nn.Module, batch_size: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    # Replace non-finite on-policy states and project the result back into the domain
    def sanitize_onpolicy_states(self, onpolicy_states: torch.Tensor, device: torch.device) -> torch.Tensor:
        x = onpolicy_states.detach().clone()
        finite_row = torch.isfinite(x).all(dim=1)
        if not finite_row.all():
            num_bad = int((~finite_row).sum().item())
            x[~finite_row] = self.sample_cover_states(num_bad, device)
        return self.project(x)

    # Build the anchor law nu_X 
    def mix_anchors(self, onpolicy_states: torch.Tensor, device: torch.device) -> torch.Tensor:
        x_on = self.sanitize_onpolicy_states(onpolicy_states, device)
        batch_size = x_on.shape[0]
        num_cover = int(self.cfg.rho_cover * batch_size)
        if num_cover <= 0:
            return x_on
        num_on = batch_size - num_cover
        perm = torch.randperm(batch_size, device=device)
        return torch.cat(
            [x_on[perm[:num_on]], self.sample_cover_states(num_cover, device)],
            dim=0,
        )


# CPU rollout interface
# Note: replace this when you change the environment used for on-policy data collection
class RolloutEnvBase:

    # Reset the rollout state to a fresh initial condition
    def reset(self) -> np.ndarray:
        raise NotImplementedError

    # Advance one rollout step and return next_state, one-step cost, and done
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        raise NotImplementedError


# GPU-side model interface
# These functions feed the Hamiltonian terms in eqs (3.1) and (2.7)
class ProblemModelGPUBase:

    # Drift term f(x,u) in the controlled diffusion
    def drift(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # Running cost ell(x,u) from the control objective
    def running_cost(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # Diffusion contribution 1/2 tr(a(x,u) A)
    def tr_aA_half(self, x: torch.Tensor, u: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# This is to It keeps the replaceable environment logic confined
class ProblemSpec:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.domain: DomainAdapter

    # Build the CPU rollout environment used by the worker processes
    def make_cpu_env(self, seed: int) -> RolloutEnvBase:
        raise NotImplementedError

    # Build the GPU model used by the Hamiltonian and viscosity losses
    def make_gpu_model(self, device: torch.device) -> ProblemModelGPUBase:
        raise NotImplementedError



########## Default domain (annulus used by the Euler experiment) ##########
# Projection Pi for the shipped Euler annulus
# This is the projection used by the prox-opt stationarity penalty
def project_to_annulus(x: torch.Tensor, r_inner: float, r_outer: float, eps: float = 1e-8) -> torch.Tensor:
    
    # Euclidean projection onto the closed annulus used by the Euler example
    if r_inner < 0.0:
        raise ValueError(f"r_inner must be nonnegative, got {r_inner}")
    if r_outer <= 0.0:
        raise ValueError(f"r_outer must be positive, got {r_outer}")
    if r_inner > r_outer:
        raise ValueError(f"Require r_inner <= r_outer, got {r_inner} > {r_outer}")

    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    safe_norms = norms.clamp_min(eps)
    dirs = x / safe_norms

    e1 = torch.zeros_like(x)
    e1[..., 0] = 1.0
    dirs = torch.where(norms > eps, dirs, e1)

    proj_inner = dirs * float(r_inner)
    proj_outer = dirs * float(r_outer)

    y = x
    y = torch.where(norms < float(r_inner), proj_inner, y)
    y = torch.where(norms > float(r_outer), proj_outer, y)
    return y


# Domain adapter for the Euler example
# You may replace swap this class without touching the method core
class EulerAnnulusDomain(DomainAdapter):

    # Project a batch of states back to the closed domain
    def project(self, x: torch.Tensor) -> torch.Tensor:
        return project_to_annulus(x, float(self.cfg.target_radius), float(self.cfg.domain_radius))

    # Sample anchor states from the covering distribution nu_cov
    def sample_cover_states(self, num: int, device: torch.device) -> torch.Tensor:
        dim = self.cfg.state_dim
        dirs = torch.randn(num, dim, device=device)
        dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
        u = torch.rand(num, 1, device=device)
        r_inner = float(self.cfg.target_radius)
        r_outer = float(self.cfg.domain_radius)
        radii = (r_inner ** dim + u * (r_outer ** dim - r_inner ** dim)).pow(1.0 / dim)
        return radii * dirs

    # Mark contact points that lie on the boundary
    # The hinge is set to zero
    def boundary_contact_mask(self, z: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.norm(z, dim=-1)
        r_inner = float(self.cfg.target_radius)
        r_outer = float(self.cfg.domain_radius)
        tol = 10.0 * torch.finfo(z.dtype).eps * max(1.0, r_outer)
        on_inner = torch.abs(norms - r_inner) <= tol
        on_outer = torch.abs(norms - r_outer) <= tol
        return on_inner | on_outer

    # Sample boundary or target points for the critic boundary penalty
    def boundary_penalty(self, critic: nn.Module, batch_size: int, device: torch.device) -> torch.Tensor:
        dim = self.cfg.state_dim
        dirs = torch.randn(batch_size, dim, device=device)
        dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
        u = torch.rand(batch_size, 1, device=device)
        radii = self.cfg.target_radius * u.pow(1.0 / dim)
        x_target = radii * dirs
        return critic(x_target).pow(2).mean()



########## CPU rollout environment for the Euler rigid body ##########
# Note: this is example-specific and is only used to generate on-policy trajectory data
class EulerRigidBodyEnvCPU(RolloutEnvBase):
    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.I = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.Q_diag = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.R_diag = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        self.sigma = cfg.sde_noise_scale * np.ones(3, dtype=np.float32)
        self.domain_radius = float(cfg.domain_radius)
        self.target_radius = float(cfg.target_radius)
        self.state: Optional[np.ndarray] = None
        self.ep_len = 0

    # Reset the rollout state to a new initial condition
    def reset(self) -> np.ndarray:
        self.ep_len = 0
        r_min = max(float(self.cfg.init_radius_min), float(self.target_radius))
        r_max = min(float(self.cfg.init_radius_max), float(self.domain_radius))
        if r_max <= r_min:
            raise ValueError(
                f"Invalid initialization shell [{r_min}, {r_max}] for target_radius={self.target_radius} and domain_radius={self.domain_radius}."
            )

        dim = int(self.cfg.state_dim)
        while True:
            direction = self.rng.standard_normal(size=(dim,)).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-12
            u = float(self.rng.random())
            radius = (r_min ** dim + u * (r_max ** dim - r_min ** dim)) ** (1.0 / dim)
            x = (radius * direction).astype(np.float32)
            norm_x = float(np.linalg.norm(x))
            if self.target_radius < norm_x < self.domain_radius:
                self.state = x
                return x.copy()

    # Check whether the state is in the target set
    def in_target(self, x: np.ndarray) -> bool:
        return float(np.linalg.norm(x)) <= self.target_radius

    # Check whether the state is still inside the outer domain
    def in_domain(self, x: np.ndarray) -> bool:
        return float(np.linalg.norm(x)) <= self.domain_radius

    # Euler rigid-body drift used by the rollout environment
    def drift(self, omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
        I1, I2, I3 = self.I[0], self.I[1], self.I[2]
        w1, w2, w3 = omega[0], omega[1], omega[2]
        t1, t2, t3 = tau[0], tau[1], tau[2]
        f1 = ((I2 - I3) / I1) * w2 * w3 + t1 / I1
        f2 = ((I3 - I1) / I2) * w3 * w1 + t2 / I2
        f3 = ((I1 - I2) / I3) * w1 * w2 + t3 / I3
        return np.array([f1, f2, f3], dtype=np.float32)

    # One-step running cost ℓ(omega, u) for the Euler example
    def running_cost(self, omega: np.ndarray, tau: np.ndarray) -> float:
        x_cost = float(np.sum(self.Q_diag * (omega ** 2)))
        u_cost = float(np.sum(self.R_diag * (tau ** 2)))
        return x_cost + u_cost

    # Advance one rollout step and return next_state, one-step cost and done flag
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        if self.state is None:
            raise RuntimeError("Environment state is not initialized. Call reset() first.")

        dt = float(self.cfg.dt)
        omega = self.state
        tau = action.astype(np.float32)

        noise = self.rng.standard_normal(size=(3,)).astype(np.float32)
        omega_next = omega + self.drift(omega, tau) * dt + self.sigma * math.sqrt(dt) * noise

        cost = self.running_cost(omega, tau) * dt
        self.state = omega_next.astype(np.float32)
        self.ep_len += 1

        hit_target = self.in_target(self.state)
        out_domain = not self.in_domain(self.state)
        time_up = self.ep_len >= int(self.cfg.max_episode_steps)
        done = bool(hit_target or out_domain or time_up)

        terminal_penalty = 0.0
        if out_domain and self.cfg.terminal_exit_penalty_coef > 0.0:
            overshoot = max(0.0, float(np.linalg.norm(self.state) - self.domain_radius))
            terminal_penalty = float(self.cfg.terminal_exit_penalty_coef) * (1.0 + overshoot ** 2)

        step_cost = cost + terminal_penalty
        if done:
            next_state = self.reset()
        else:
            next_state = self.state.copy()
        return next_state, float(step_cost), done


########## GPU problem model for the Euler rigid body ##########
class EulerRigidBodyModelGPU(ProblemModelGPUBase):
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.I = torch.tensor([1.0, 2.0, 3.0], device=device)
        self.Q_diag = torch.tensor([1.0, 1.0, 1.0], device=device)
        self.R_diag = torch.tensor([0.1, 0.1, 0.1], device=device)
        self.sigma = cfg.sde_noise_scale * torch.ones(3, device=device)
        self.sigma_sq = self.sigma ** 2

    # Euler drift f(omega,u) on the GPU side
    def drift(self, omega: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        I1, I2, I3 = self.I[0], self.I[1], self.I[2]
        w1, w2, w3 = omega[..., 0], omega[..., 1], omega[..., 2]
        t1, t2, t3 = tau[..., 0], tau[..., 1], tau[..., 2]
        f1 = ((I2 - I3) / I1) * w2 * w3 + t1 / I1
        f2 = ((I3 - I1) / I2) * w3 * w1 + t2 / I2
        f3 = ((I1 - I2) / I3) * w1 * w2 + t3 / I3
        return torch.stack([f1, f2, f3], dim=-1)

    # Running cost ℓ(omega,u) on the GPU side
    def running_cost(self, omega: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x_cost = torch.sum(self.Q_diag * (omega ** 2), dim=-1)
        u_cost = torch.sum(self.R_diag * (tau ** 2), dim=-1)
        return x_cost + u_cost

    # Diffusion contribution 1/2 tr(aA) for the Euler example
    def tr_aA_half(self, omega: torch.Tensor, tau: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        diag_A = torch.diagonal(A, dim1=-2, dim2=-1)
        return 0.5 * torch.sum(self.sigma_sq * diag_A, dim=-1)


# Put together the default Euler domain, CPU rollout env, and GPU model
class EulerRigidBodyProblem(ProblemSpec):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.domain = EulerAnnulusDomain(cfg)

    # Build the CPU rollout environment used by the worker processes
    def make_cpu_env(self, seed: int) -> RolloutEnvBase:
        return EulerRigidBodyEnvCPU(self.cfg, seed)

    # Build the GPU model used by the Hamiltonian and viscosity losses
    def make_gpu_model(self, device: torch.device) -> ProblemModelGPUBase:
        return EulerRigidBodyModelGPU(self.cfg, device)


# If you change the environment, this is the function you should edit
def build_problem(cfg: Config) -> ProblemSpec:
    # Swap this return value when you replace the Euler example
    # Note: A new problem would needs a DomainAdapter, a CPU rollout env,
    # a GPU model, and a ProblemSpec that wires them together.
    return EulerRigidBodyProblem(cfg)