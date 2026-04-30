import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from problem import ProblemModelGPUBase, ProblemSpec
from networks import SquashedGaussianPolicy


########## METHOD CORE ##########

# Samples the SPD curvature bank M from eq (3.3)
def sample_spd_bank(num_mats: int, dim: int, alpha_min: float, alpha_max: float, device: torch.device) -> torch.Tensor:
    log_alpha = torch.empty(num_mats, dim, device=device).uniform_(math.log10(alpha_min), math.log10(alpha_max))
    alpha = 10.0 ** log_alpha
    D = torch.diag_embed(alpha)
    A = torch.randn(num_mats, dim, dim, device=device)
    Q, R = torch.linalg.qr(A)
    diag_R = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    Q = Q * diag_R.unsqueeze(-2)
    return Q.transpose(-2, -1) @ D @ Q


# Builds the prox-generated contact points and their envelope jets
# Note: zeta^-, zeta^+, p^-, p^+, A^-, A^+ are used in all PDE-side losses
def compute_contacts_and_jets(
    prox: nn.Module,
    anchors: torch.Tensor,
    M_bank: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = anchors.device
    batch_size, dim = anchors.shape
    num_mats = M_bank.shape[0]
    total = batch_size * num_mats

    x_rep = anchors[:, None, :].expand(batch_size, num_mats, dim).reshape(total, dim)
    M_rep = M_bank[None, :, :, :].expand(batch_size, num_mats, dim, dim).reshape(total, dim, dim)

    b_minus = -torch.ones(total, 1, device=device)
    b_plus = torch.ones(total, 1, device=device)

    x_both = torch.cat([x_rep, x_rep], dim=0)
    M_both = torch.cat([M_rep, M_rep], dim=0)
    b_both = torch.cat([b_minus, b_plus], dim=0)
    z_both = prox(x_both, M_both, b_both)

    z_minus = z_both[:total]
    z_plus = z_both[total:]

    diff_minus = (x_rep - z_minus).unsqueeze(-1)
    diff_plus = (x_rep - z_plus).unsqueeze(-1)
    p_minus = torch.bmm(M_rep, diff_minus).squeeze(-1)
    p_plus = -torch.bmm(M_rep, diff_plus).squeeze(-1)
    A_minus = -M_rep
    A_plus = M_rep
    return x_rep, z_minus, z_plus, p_minus, p_plus, A_minus, A_plus


# Computes the policy-conditioned operator: eq (3.1)
def policy_conditioned_hamiltonian(
    model: ProblemModelGPUBase,
    beta: float,
    V: torch.Tensor,
    p: torch.Tensor,
    A: torch.Tensor,
    z: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    ell = model.running_cost(z, u)
    drift = model.drift(z, u)
    pf = torch.sum(p * drift, dim=-1)
    tr_term = model.tr_aA_half(z, u, A)
    return beta * V - (ell + pf + tr_term)



# Evaluates the control Hamiltonian inner term
def hjb_inner_term(
    model: ProblemModelGPUBase,
    z: torch.Tensor,
    u: torch.Tensor,
    p: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    ell = model.running_cost(z, u)
    drift = model.drift(z, u)
    pf = torch.sum(p * drift, dim=-1)
    tr_term = model.tr_aA_half(z, u, A)
    return ell + pf + tr_term


# Critic-side viscosity penalty L_Visc
# Note: we use the hinge violations g_super and g_sub from the viscosity-loss, eq (3.4) 
def viscosity_loss(
    problem: ProblemSpec,
    cfg: Config,
    model: ProblemModelGPUBase,
    actor: SquashedGaussianPolicy,
    critic: nn.Module,
    prox: nn.Module,
    anchors: torch.Tensor,
    M_bank: torch.Tensor,
) -> torch.Tensor:
    batch_size = anchors.shape[0]
    num_mats = M_bank.shape[0]

    with torch.no_grad():
        _, z_minus, z_plus, p_minus, p_plus, A_minus, A_plus = compute_contacts_and_jets(prox, anchors, M_bank)

    V_all = critic(torch.cat([z_minus, z_plus], dim=0))
    total = z_minus.shape[0]
    V_minus = V_all[:total]
    V_plus = V_all[total:]

    with torch.no_grad():
        u_minus = actor.deterministic_action(z_minus)
        u_plus = actor.deterministic_action(z_plus)

    H_super = policy_conditioned_hamiltonian(model, cfg.beta, V_minus, p_minus, A_minus, z_minus, u_minus)
    H_sub = policy_conditioned_hamiltonian(model, cfg.beta, V_plus, p_plus, A_plus, z_plus, u_plus)

    boundary_minus = problem.domain.boundary_contact_mask(z_minus).reshape(batch_size, num_mats)
    boundary_plus = problem.domain.boundary_contact_mask(z_plus).reshape(batch_size, num_mats)

    g_super = (-H_super).reshape(batch_size, num_mats).masked_fill(boundary_minus, 0.0)
    g_sub = H_sub.reshape(batch_size, num_mats).masked_fill(boundary_plus, 0.0)

    g_super_max = g_super.max(dim=1).values
    g_sub_max = g_sub.max(dim=1).values
    return (F.relu(g_super_max).pow(2) + F.relu(g_sub_max).pow(2)).mean()


# Computes the inf-envelope and sup-envelope energies E_inf and E_sup
# Note: these are from the regularizer L_env in the prox objective
def envelope_energies_from_V(
    V_minus: torch.Tensor,
    V_plus: torch.Tensor,
    x_rep: torch.Tensor,
    z_minus: torch.Tensor,
    z_plus: torch.Tensor,
    p_minus: torch.Tensor,
    p_plus: torch.Tensor,
    batch_size: int,
    num_mats: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    diff_minus = x_rep - z_minus
    diff_plus = x_rep - z_plus
    quad_minus = torch.sum(diff_minus * p_minus, dim=-1)
    quad_plus = torch.sum(diff_plus * (-p_plus), dim=-1)
    E_inf = (V_minus + 0.5 * quad_minus).reshape(batch_size, num_mats)
    E_sup = (V_plus - 0.5 * quad_plus).reshape(batch_size, num_mats)
    return E_inf, E_sup


# Copmpute L_env by taking the envelope energies at the same worst-case curvature
# Note: we use indices that maximize g_super and g_sub for each anchor
def L_env_worstcase_aligned(E_inf: torch.Tensor, E_sup: torch.Tensor, g_super: torch.Tensor, g_sub: torch.Tensor) -> torch.Tensor:
    batch_size = g_super.shape[0]
    arange = torch.arange(batch_size, device=g_super.device)
    k_super = g_super.argmax(dim=1)
    k_sub = g_sub.argmax(dim=1)
    return (E_inf[arange, k_super] - E_sup[arange, k_sub]).mean()


# Computes L_prox-opt (KKT stationarity condition)
def prox_opt_loss_from_V(
    problem: ProblemSpec,
    cfg: Config,
    V_minus: torch.Tensor,
    V_plus: torch.Tensor,
    z_minus: torch.Tensor,
    z_plus: torch.Tensor,
    p_minus: torch.Tensor,
    p_plus: torch.Tensor,
) -> torch.Tensor:
    grad_minus = torch.autograd.grad(V_minus.sum(), z_minus, create_graph=True)[0]
    grad_plus = torch.autograd.grad(V_plus.sum(), z_plus, create_graph=True)[0]

    grad_phi_minus = grad_minus - p_minus
    grad_phi_plus = p_plus - grad_plus
    eta = 1.0 / (1.0 + float(cfg.alpha_max))

    y_minus = z_minus - eta * grad_phi_minus
    y_plus = z_plus - eta * grad_phi_plus
    proj_minus = problem.domain.project(y_minus)
    proj_plus = problem.domain.project(y_plus)

    G_minus = (z_minus - proj_minus) / eta
    G_plus = (z_plus - proj_plus) / eta
    return (torch.sum(G_minus ** 2, dim=-1) + torch.sum(G_plus ** 2, dim=-1)).mean()


# L_Prox: adversarial hinge maximization plus L_env plus L_prox-opt
def proximal_loss(
    problem: ProblemSpec,
    cfg: Config,
    model: ProblemModelGPUBase,
    actor: SquashedGaussianPolicy,
    critic: nn.Module,
    prox: nn.Module,
    anchors: torch.Tensor,
    M_bank: torch.Tensor,
) -> torch.Tensor:
    batch_size = anchors.shape[0]
    num_mats = M_bank.shape[0]

    x_rep, z_minus, z_plus, p_minus, p_plus, A_minus, A_plus = compute_contacts_and_jets(prox, anchors, M_bank)

    V_all = critic(torch.cat([z_minus, z_plus], dim=0))
    total = z_minus.shape[0]
    V_minus = V_all[:total]
    V_plus = V_all[total:]

    u_minus = actor.deterministic_action(z_minus)
    u_plus = actor.deterministic_action(z_plus)

    H_super = policy_conditioned_hamiltonian(model, cfg.beta, V_minus, p_minus, A_minus, z_minus, u_minus)
    H_sub = policy_conditioned_hamiltonian(model, cfg.beta, V_plus, p_plus, A_plus, z_plus, u_plus)

    boundary_minus = problem.domain.boundary_contact_mask(z_minus).reshape(batch_size, num_mats)
    boundary_plus = problem.domain.boundary_contact_mask(z_plus).reshape(batch_size, num_mats)

    g_super = (-H_super).reshape(batch_size, num_mats).masked_fill(boundary_minus, 0.0)
    g_sub = H_sub.reshape(batch_size, num_mats).masked_fill(boundary_plus, 0.0)

    g_super_max = g_super.max(dim=1).values
    g_sub_max = g_sub.max(dim=1).values
    adv_term = (g_super_max + g_sub_max).mean()
    loss_adv = -cfg.lambda_adv * adv_term

    if cfg.lambda_env > 0.0:
        E_inf, E_sup = envelope_energies_from_V(
            V_minus,
            V_plus,
            x_rep,
            z_minus,
            z_plus,
            p_minus,
            p_plus,
            batch_size=batch_size,
            num_mats=num_mats,
        )
        L_env = L_env_worstcase_aligned(E_inf, E_sup, g_super, g_sub)
    else:
        L_env = torch.zeros((), device=anchors.device)

    L_prox_opt = prox_opt_loss_from_V(problem, cfg, V_minus, V_plus, z_minus, z_plus, p_minus, p_plus)
    return loss_adv + cfg.lambda_env * L_env + cfg.lambda_prox_opt * L_prox_opt


# L_jet: for pushing the policy toward greedy Hamiltonian minimizers
def jet_alignment_loss(
    problem: ProblemSpec,
    cfg: Config,
    model: ProblemModelGPUBase,
    actor: SquashedGaussianPolicy,
    prox: nn.Module,
    anchors: torch.Tensor,
    M_bank: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        _, z_minus, z_plus, p_minus, p_plus, A_minus, A_plus = compute_contacts_and_jets(prox, anchors, M_bank)

    z_minus = z_minus.detach()
    z_plus = z_plus.detach()
    p_minus = p_minus.detach()
    p_plus = p_plus.detach()
    A_minus = A_minus.detach()
    A_plus = A_plus.detach()

    u_minus = actor.deterministic_action(z_minus)
    u_plus = actor.deterministic_action(z_plus)

    boundary_minus = problem.domain.boundary_contact_mask(z_minus)
    boundary_plus = problem.domain.boundary_contact_mask(z_plus)

    inner_minus = hjb_inner_term(model, z_minus, u_minus, p_minus, A_minus).masked_fill(boundary_minus, 0.0)
    inner_plus = hjb_inner_term(model, z_plus, u_plus, p_plus, A_plus).masked_fill(boundary_plus, 0.0)
    return inner_minus.mean() + inner_plus.mean()



########## On-policy cost targets ##########
# Notes: computes critic targets and actor PPO advantages from one-step costs
# This is the discrete-time cost recursion used by the on-policy update
# Actor advantages are the sign-flipped cost advantages
def compute_gae_cost_returns_and_actor_advantages(
    cfg: Config,
    step_costs: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # GAE written in cost language, consistent with the paper.
    # The critic targets discounted cost-to-go, and the actor uses the sign-flipped
    # cost advantages inside the PPO objective.
    num_envs, horizon = step_costs.shape
    gamma = math.exp(-cfg.beta * cfg.dt)
    lam = cfg.gae_lambda

    advantages_cost = torch.zeros_like(step_costs)
    gae = torch.zeros(num_envs, device=step_costs.device)
    next_value = last_values

    for t in reversed(range(horizon)):
        mask = 1.0 - dones[:, t]
        delta = step_costs[:, t] + gamma * next_value * mask - values[:, t]
        gae = delta + gamma * lam * mask * gae
        advantages_cost[:, t] = gae
        next_value = values[:, t]

    returns_cost = advantages_cost + values
    actor_advantages = -advantages_cost
    return returns_cost, actor_advantages