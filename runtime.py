import os
import socket
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config import Config
from networks import SquashedGaussianPolicy
from problem import build_problem


########## Runtime stuff ##########
# Note: we need this for multi-GPU functionality
# We pick a local TCP port for single-node DDP startup
def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


# Initializes the NCCL process group for multi-GPU training
def ddp_setup(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


# Tears down the NCCL process group
def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# Splits the total rollout workers evenly across ranks so every rank has the same load
def compute_equal_worker_allocation(total_workers: int, world_size: int) -> Tuple[int, int]:
    total_used = (total_workers // world_size) * world_size
    per_rank = total_used // world_size if world_size > 0 else 0
    return per_rank, total_used


# Actor gradients are synchronized manually instead of wrapping the actor in DDP
# This performs that all-reduce in one flat buffer
class FusedGradAverager:

    def __init__(self, params: List[torch.nn.Parameter], world_size: int, device: torch.device):
        self.params = [p for p in params if p.requires_grad]
        self.numels = [p.numel() for p in self.params]
        self.total = int(sum(self.numels))
        self.world_size = float(world_size)
        self.buffer = torch.empty(self.total, device=device, dtype=torch.float32)

    @torch.no_grad()
    def allreduce_mean_(self) -> None:
        offset = 0
        for p, numel in zip(self.params, self.numels):
            if p.grad is None:
                self.buffer[offset : offset + numel].zero_()
            else:
                grad = p.grad
                if grad.dtype != torch.float32:
                    self.buffer[offset : offset + numel].copy_(grad.float().view(-1))
                else:
                    self.buffer[offset : offset + numel].copy_(grad.view(-1))
            offset += numel

        dist.all_reduce(self.buffer, op=dist.ReduceOp.SUM)
        self.buffer.div_(self.world_size)

        offset = 0
        for p, numel in zip(self.params, self.numels):
            if p.grad is not None:
                view = self.buffer[offset : offset + numel].view_as(p.grad)
                if p.grad.dtype != torch.float32:
                    p.grad.copy_(view.to(dtype=p.grad.dtype))
                else:
                    p.grad.copy_(view)
            offset += numel


# Averages a stack of scalar diagnostics across ranks
def reduce_mean_stack(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= float(dist.get_world_size())
    return y


# CPU rollout worker loop
# It receives the latest actor weights, collects trajectories, and sends back a batch
def rollout_worker_loop(conn, cfg: Config, global_worker_id: int) -> None:
    torch.set_num_threads(1)
    np.random.seed(cfg.seed + 1000 * global_worker_id)
    torch.manual_seed(cfg.seed + 1000 * global_worker_id)

    # Each worker keeps a CPU copy of the actor and refreshes it before every rollout
    actor = SquashedGaussianPolicy(cfg, cfg.hidden_dims).cpu()
    actor.eval()

    problem = build_problem(cfg)
    env = problem.make_cpu_env(seed=cfg.seed + 2000 * global_worker_id)
    state = env.reset()

    while True:
        msg = conn.recv()
        cmd = msg["cmd"]
        if cmd == "close":
            break
        if cmd != "rollout":
            raise RuntimeError(f"Unknown worker command: {cmd}")

        actor.load_state_dict(msg["actor_state_dict"])
        actor.eval()
        steps = int(msg["steps"])

        states = np.zeros((steps, cfg.state_dim), dtype=np.float32)
        actions = np.zeros((steps, cfg.action_dim), dtype=np.float32)
        logps = np.zeros((steps,), dtype=np.float32)
        step_costs = np.zeros((steps,), dtype=np.float32)
        dones = np.zeros((steps,), dtype=np.bool_)

        for t in range(steps):
            states[t] = state
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            action_t, logp_t = actor.sample_action_and_logp(state_t)
            action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)

            next_state, step_cost, done = env.step(action_np)
            actions[t] = action_np
            logps[t] = float(logp_t.item())
            step_costs[t] = step_cost
            dones[t] = done
            state = next_state

        conn.send(
            {
                "states": states,
                "actions": actions,
                "logps": logps,
                "step_costs": step_costs,
                "dones": dones,
                "last_state": state.copy(),
            }
        )


# Starts the rollout workers assigned to the current rank
def start_rollout_workers(cfg: Config, global_worker_ids: List[int]):
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for gid in global_worker_ids:
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=rollout_worker_loop, args=(child_conn, cfg, gid))
        proc.daemon = True
        proc.start()
        conns.append(parent_conn)
        procs.append(proc)
    return conns, procs


# Stops and joins the rollout workers
def stop_rollout_workers(conns, procs) -> None:
    for conn in conns:
        try:
            conn.send({"cmd": "close"})
        except Exception:
            pass
    for proc in procs:
        proc.join(timeout=5)