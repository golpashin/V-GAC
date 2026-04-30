from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from config import Config
from problem import DomainAdapter


########## Netwroks and shared utilities ##########
# Note: here we define the actor pi_phi, critic V_theta, prox network P_psi

# Helper for the alternating prox --> critic --> actor updates
def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)

# Stable inverse tanh used for evaluating squashed-Gaussian log-probabilities
def atanh_stable(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


# Weight-normalized MLP
# NNote: this is shared by the actor, critic, and prox networks
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(weight_norm(nn.Linear(last_dim, hidden_dim)))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(weight_norm(nn.Linear(last_dim, out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Actor network pi_phi
# Note: rollout is stochastic, but the PDE-side losses use deterministic mean action
class SquashedGaussianPolicy(nn.Module):

    def __init__(self, cfg: Config, hidden_dims: Tuple[int, ...]):
        super().__init__()
        self.cfg = cfg
        self.backbone = MLP(cfg.state_dim, cfg.action_dim, hidden_dims)
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))
        self.action_limit = float(cfg.action_limit)
        self.log_std_min = float(cfg.log_std_min)
        self.log_std_max = float(cfg.log_std_max)
        self.eps = float(cfg.tanh_eps)
        self.mu_clip = float(cfg.policy_mu_clip)

    def _get_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.backbone(x)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=self.mu_clip, neginf=-self.mu_clip)
        mu = torch.clamp(mu, -self.mu_clip, self.mu_clip)

        log_std = torch.nan_to_num(
            self.log_std,
            nan=0.0,
            posinf=self.log_std_max,
            neginf=self.log_std_min,
        )
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mu)
        return mu, log_std.expand_as(mu), std

    # Deterministic feedback action used in the PDE-side losses
    def deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        mu, _, _ = self._get_params(x)
        return self.action_limit * torch.tanh(mu)

    @torch.no_grad()
    # Sample a rollout action from the squashed Gaussian policy and return its log-density
    def sample_action_and_logp(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, _, std = self._get_params(x)
        distn = torch.distributions.Normal(mu, std)
        u = distn.sample()
        a = self.action_limit * torch.tanh(u)
        logp_u = distn.log_prob(u).sum(dim=-1)
        y = torch.tanh(u)
        log_det = torch.log(self.action_limit * (1.0 - y * y) + self.eps).sum(dim=-1)
        return a, logp_u - log_det

    # Evaluate the log-density needed by the PPO ratio
    def log_prob(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mu, _, std = self._get_params(x)
        distn = torch.distributions.Normal(mu, std)
        y = torch.clamp(a / self.action_limit, -1.0 + self.eps, 1.0 - self.eps)
        u = atanh_stable(y)
        logp_u = distn.log_prob(u).sum(dim=-1)
        log_det = torch.log(self.action_limit * (1.0 - y * y) + self.eps).sum(dim=-1)
        return logp_u - log_det

    # Entropy term used in the PPO actor objective
    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        mu, _, std = self._get_params(x)
        distn = torch.distributions.Normal(mu, std)
        base_entropy = distn.entropy().sum(dim=-1)
        u = distn.rsample()
        y = torch.tanh(u)
        log_det = torch.log(self.action_limit * (1.0 - y * y) + self.eps).sum(dim=-1)
        return (base_entropy + log_det).mean()


# Critic network V_theta used in the TD term and the viscosity loss
class Critic(nn.Module):
    def __init__(self, cfg: Config, hidden_dims: Tuple[int, ...]):
        super().__init__()
        self.net = MLP(cfg.state_dim, 1, hidden_dims)

    # Evaluate V_theta(x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# Prox network P_psi that gives the contact points zeta^- and zeta^+
class ProximalNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: Tuple[int, ...], domain: DomainAdapter):
        super().__init__()
        self.domain = domain
        in_dim = state_dim + state_dim * state_dim + 1
        self.net = MLP(in_dim, state_dim, hidden_dims)

    # Map an anchor, curvature, and polarity to a contact point
    # Then project it back into the domain
    def forward(self, x: torch.Tensor, M: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n, dim = x.shape
        inputs = torch.cat([x, M.reshape(n, dim * dim), b], dim=-1)
        raw = self.net(inputs)
        return self.domain.project(raw)