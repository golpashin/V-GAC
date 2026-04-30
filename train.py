from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from config import Config
from method import (
    compute_gae_cost_returns_and_actor_advantages,
    jet_alignment_loss,
    proximal_loss,
    sample_spd_bank,
    viscosity_loss,
)
from networks import Critic, ProximalNet, SquashedGaussianPolicy, set_requires_grad
from problem import build_problem
from runtime import (
    FusedGradAverager,
    compute_equal_worker_allocation,
    ddp_cleanup,
    ddp_setup,
    find_free_port,
    reduce_mean_stack,
    start_rollout_workers,
    stop_rollout_workers,
)


########## Training loop ##########
# Notes: here we collect trajectories, build the anchor set, then alternate
# prox, critic, and actor updates on each minibatch.
# This is single-rank training routine.
# Each iteration collects on-policy data, then alternates prox, critic,
# and actor updates in the same order
def train_worker(rank: int, world_size: int, cfg: Config, master_port: int) -> None:
    torch.set_num_threads(1)
    ddp_setup(rank, world_size, master_port)

    local_rank = rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    problem = build_problem(cfg)
    model = problem.make_gpu_model(device)

    actor = SquashedGaussianPolicy(cfg, cfg.hidden_dims).to(device)
    critic = Critic(cfg, cfg.hidden_dims).to(device)
    prox = ProximalNet(cfg.state_dim, cfg.hidden_dims, problem.domain).to(device)

    actor_grad_averager = FusedGradAverager(list(actor.parameters()), world_size, device)
    ddp_critic = DDP(critic, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True, find_unused_parameters=False)
    ddp_prox = DDP(prox, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    torch.manual_seed(cfg.seed + 12345 * rank)
    np.random.seed(cfg.seed + 12345 * rank)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_critic = torch.optim.Adam(ddp_critic.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay)
    opt_prox = torch.optim.Adam(ddp_prox.parameters(), lr=cfg.lr_prox)

    workers_per_rank, total_used_workers = compute_equal_worker_allocation(cfg.num_rollout_workers, world_size)
    if workers_per_rank <= 0:
        raise RuntimeError(
            f"num_rollout_workers={cfg.num_rollout_workers} is too small for world_size={world_size}."
        )

    if rank == 0:
        print(
            f"V-GAC training | problem={problem.__class__.__name__} | GPUs={world_size} | workers={total_used_workers} ({workers_per_rank}/rank) | steps_per_worker={cfg.steps_per_worker}",
            flush=True,
        )
        print(
            "To replace the environment, edit SECTION 2 and change build_problem(cfg).",
            flush=True,
        )
        if total_used_workers != cfg.num_rollout_workers:
            print(
                f"Requested {cfg.num_rollout_workers} rollout workers; using {total_used_workers} so every rank has the same worker count.",
                flush=True,
            )

    start_gid = rank * workers_per_rank
    worker_ids = list(range(start_gid, start_gid + workers_per_rank))
    conns, procs = start_rollout_workers(cfg, worker_ids)

    env_steps_total = 0
    # Counts actual terminal events (done=True) seen by the rollout workers
    # With long exit-time horizons this can stay zero for many iterations
    completed_episodes_total = 0

    try:
        for iteration in range(cfg.total_iterations):
            actor_state_cpu = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
            for conn in conns:
                conn.send({"cmd": "rollout", "actor_state_dict": actor_state_cpu, "steps": cfg.steps_per_worker})
            worker_rollouts = [conn.recv() for conn in conns]

            num_envs_local = workers_per_rank
            horizon = cfg.steps_per_worker
            state_dim = cfg.state_dim
            action_dim = cfg.action_dim

            if rank == 0:
                env_steps_total += num_envs_local * horizon * world_size

            completed_episodes_local = int(sum(np.asarray(r["dones"], dtype=np.int32).sum() for r in worker_rollouts))
            mean_step_cost_local = float(np.mean([np.mean(r["step_costs"]) for r in worker_rollouts])) if worker_rollouts else 0.0

            summary_tensor = torch.tensor([float(completed_episodes_local), mean_step_cost_local], device=device)
            dist.all_reduce(summary_tensor, op=dist.ReduceOp.SUM)
            completed_episodes_global = int(summary_tensor[0].item())
            mean_step_cost_global = float(summary_tensor[1].item() / world_size)

            if rank == 0:
                completed_episodes_total += completed_episodes_global

            states_np = np.stack([r["states"] for r in worker_rollouts], axis=0)
            actions_np = np.stack([r["actions"] for r in worker_rollouts], axis=0)
            logps_np = np.stack([r["logps"] for r in worker_rollouts], axis=0)
            step_costs_np = np.stack([r["step_costs"] for r in worker_rollouts], axis=0)
            dones_np = np.stack([r["dones"] for r in worker_rollouts], axis=0)
            last_states_np = np.stack([r["last_state"] for r in worker_rollouts], axis=0)

            states = torch.tensor(states_np, device=device, dtype=torch.float32)
            actions = torch.tensor(actions_np, device=device, dtype=torch.float32)
            logps_old = torch.tensor(logps_np, device=device, dtype=torch.float32)
            step_costs = torch.tensor(step_costs_np, device=device, dtype=torch.float32)
            dones = torch.tensor(dones_np.astype(np.float32), device=device, dtype=torch.float32)
            last_states = torch.tensor(last_states_np, device=device, dtype=torch.float32)

            with torch.no_grad():
                values = ddp_critic(states.reshape(num_envs_local * horizon, state_dim)).reshape(num_envs_local, horizon)
                last_values = ddp_critic(last_states)

            returns_cost, actor_advantages = compute_gae_cost_returns_and_actor_advantages(cfg, step_costs, dones, values, last_values)

            flat_states = states.reshape(num_envs_local * horizon, state_dim)
            flat_actions = actions.reshape(num_envs_local * horizon, action_dim)
            flat_logps_old = logps_old.reshape(num_envs_local * horizon)
            flat_returns_cost = returns_cost.reshape(num_envs_local * horizon)
            flat_advantages = actor_advantages.reshape(num_envs_local * horizon)

            effective_mb = (cfg.minibatch_size // world_size) * world_size
            local_mb = effective_mb // world_size
            if local_mb <= 0:
                raise RuntimeError(f"minibatch_size={cfg.minibatch_size} is too small for world_size={world_size}.")

            num_samples_local = flat_states.shape[0]
            effective_num_samples_local = (num_samples_local // local_mb) * local_mb
            if effective_num_samples_local <= 0:
                raise RuntimeError(
                    f"Need more local samples: N_local={num_samples_local}, local_mb={local_mb}."
                )

            last_td = torch.zeros((), device=device)
            last_visc = torch.zeros((), device=device)
            last_bdy = torch.zeros((), device=device)
            last_critic = torch.zeros((), device=device)
            last_actor = torch.zeros((), device=device)
            last_prox = torch.zeros((), device=device)

            for _ in range(cfg.ppo_epochs):
                permutation = torch.randperm(effective_num_samples_local, device=device)

                for start in range(0, effective_num_samples_local, local_mb):
                    mb_idx = permutation[start : start + local_mb]
                    batch_states = flat_states[mb_idx]
                    batch_actions = flat_actions[mb_idx]
                    batch_logp_old = flat_logps_old[mb_idx]
                    batch_returns_cost = flat_returns_cost[mb_idx]
                    batch_advantages = flat_advantages[mb_idx]

                    # Sample the curvature bank and the anchor batch for the PDE-side losses.
                    M_bank = sample_spd_bank(cfg.num_M_mats, cfg.state_dim, cfg.alpha_min, cfg.alpha_max, device)
                    anchors = problem.domain.mix_anchors(batch_states, device)

                    # Prox step: strengthen the adversary while actor and critic stay fixed.
                    set_requires_grad(ddp_critic, False)
                    set_requires_grad(actor, False)
                    set_requires_grad(ddp_prox, True)
                    for _ in range(cfg.prox_adv_steps):
                        prox_loss = proximal_loss(problem, cfg, model, actor, ddp_critic, ddp_prox, anchors, M_bank)
                        opt_prox.zero_grad(set_to_none=True)
                        prox_loss.backward()
                        torch.nn.utils.clip_grad_norm_(ddp_prox.parameters(), cfg.max_grad_norm)
                        opt_prox.step()
                        last_prox = prox_loss.detach()
                    set_requires_grad(ddp_critic, True)
                    set_requires_grad(actor, True)

                    # Critic step: TD loss + viscosity penalty + boundary penalty.
                    set_requires_grad(ddp_prox, False)
                    value_pred = ddp_critic(batch_states)
                    td_loss = F.mse_loss(value_pred, batch_returns_cost)
                    visc_loss = viscosity_loss(problem, cfg, model, actor, ddp_critic, ddp_prox, anchors, M_bank)
                    bdy_loss = problem.domain.boundary_penalty(ddp_critic, batch_states.shape[0], device)
                    critic_loss = cfg.value_coef * td_loss + cfg.lambda_visc * visc_loss + cfg.lambda_bdy * bdy_loss

                    opt_critic.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(ddp_critic.parameters(), cfg.max_grad_norm)
                    opt_critic.step()
                    set_requires_grad(ddp_prox, True)

                    # Actor step: PPO objective + entropy term + jet-alignment penalty.
                    new_logp = actor.log_prob(batch_states, batch_actions)
                    entropy = actor.entropy(batch_states)
                    log_ratio = new_logp - batch_logp_old
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_clip_eps, 1.0 + cfg.ppo_clip_eps) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    jet_loss = jet_alignment_loss(problem, cfg, model, actor, ddp_prox, anchors, M_bank)
                    actor_loss = policy_loss - cfg.entropy_coef * entropy + cfg.lambda_jet * jet_loss

                    opt_actor.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_grad_averager.allreduce_mean_()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                    opt_actor.step()

                    last_td = td_loss.detach()
                    last_visc = visc_loss.detach()
                    last_bdy = bdy_loss.detach()
                    last_critic = critic_loss.detach()
                    last_actor = actor_loss.detach()
            
            # User display
            if (iteration % cfg.display_every_iters) == 0:
                losses = reduce_mean_stack(torch.stack([last_td, last_visc, last_bdy, last_critic, last_actor, last_prox]))
                if rank == 0:
                    print(
                        f"Iter {iteration + 1:4d}/{cfg.total_iterations} | env_steps={env_steps_total} | completed_episodes={completed_episodes_total} | mean_step_cost={mean_step_cost_global:.6f} | alpha=[{cfg.alpha_min:.2e}, {cfg.alpha_max:.2e}]",
                        flush=True,
                    )
                    print(
                        "  losses: "
                        f"TD={losses[0].item():.6f} | "
                        f"Visc={losses[1].item():.6f} | "
                        f"Bdy={losses[2].item():.6f} | "
                        f"Critic={losses[3].item():.6f} | "
                        f"Actor={losses[4].item():.6f} | "
                        f"Prox={losses[5].item():.6f}",
                        flush=True,
                    )

    finally:
        stop_rollout_workers(conns, procs)
        ddp_cleanup()


# Entry point for single-node multi-GPU training
# Note: we use all visible GPUs and spawn one process per rank
def launch_training(cfg: Optional[Config] = None, world_size: Optional[int] = None) -> None:
    if cfg is None:
        cfg = Config()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    available_gpus = torch.cuda.device_count()
    if world_size is None:
        world_size = available_gpus
    world_size = int(world_size)
    if world_size < 1 or world_size > available_gpus:
        raise ValueError(f"world_size must be in [1, {available_gpus}], got {world_size}.")

    master_port = find_free_port()
    mp.spawn(train_worker, args=(world_size, cfg, master_port), nprocs=world_size, join=True)