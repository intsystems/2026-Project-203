"""
Distributed-transport test for ``signmuon_optimizers.py``.

Verifies that the sharded ``step()`` (reduce_scatter -> owning-rank update ->
all_gather, with round-robin parameter ownership, per-owner optimizer state, and
padding when ``len(params) % world_size != 0``) produces EXACTLY the same
parameters as a single-process centralized reference that calls ``update_param``
on every parameter with the mean gradient.

It runs on CPU with the gloo backend and needs no GPUs. gloo does not implement
``reduce_scatter``; we install thin SYNCHRONOUS shims for ``reduce_scatter`` and
``all_gather`` (built on gloo's ``all_reduce`` / ``all_gather``) so the *real*
optimizer code path is exercised unchanged. On a real GPU cluster the production
code uses NCCL's native async collectives instead -- the shims are test-only.

To keep the comparison exact we (a) use float64, (b) replace Newton-Schulz with a
rank-truncated exact polar factor (stable under tiny perturbations), and (c) give
every rank the identical gradient for each parameter, so reduce_scatter's average
is exact. This isolates the systems logic (ownership / padding / state / gather),
which is the part being tested; the gradient averaging itself is torch's own.

Run:  python test_distributed_sharding.py            # world_size = 4
      python test_distributed_sharding.py 2          # world_size = 2
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SIGNMUON_NO_COMPILE", "1")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

T_STEPS = 8
ATOL = 1e-9


# --- rank-stable exact polar (U_r V_r^T over nonzero singular directions) -----
def _exact_polar(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    A = G.to(torch.float64)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    if S.numel() == 0:
        return torch.zeros_like(A)
    r = int((S > 1e-9 * S[0]).sum().item())
    if r == 0:
        return torch.zeros_like(A)
    return (U[:, :r] @ Vh[:r, :]).to(G.dtype)


# --- gloo shims: synchronous reduce_scatter / all_gather returning done "work" -
class _DoneWork:
    def get_future(self):
        f = torch.futures.Future()
        f.set_result(None)
        return f

    def wait(self):
        return None


def _install_gloo_shims():
    real_all_reduce = dist.all_reduce
    real_all_gather = dist.all_gather

    def reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, async_op=False, group=None):
        ws = dist.get_world_size()
        rank = dist.get_rank()
        stacked = torch.stack([t.detach().clone() for t in input_list], dim=0).contiguous()
        real_all_reduce(stacked, op=dist.ReduceOp.SUM)   # gloo supports SUM
        res = stacked[rank]
        if op == dist.ReduceOp.AVG:
            res = res / ws
        output.copy_(res)
        return _DoneWork()

    def all_gather(tensor_list, tensor, async_op=False, group=None):
        real_all_gather(tensor_list, tensor.contiguous())
        return _DoneWork()

    dist.reduce_scatter = reduce_scatter
    dist.all_gather = all_gather


# --- deterministic, rank-independent parameters and gradients -----------------
def _make_params():
    """5 params over two shapes (3 of (5,5), 2 of (4,4)); the counts are not
    multiples of world_size in {2,4}, so padding is exercised."""
    g = torch.Generator().manual_seed(1234)
    shapes = [(5, 5), (5, 5), (5, 5), (4, 4), (4, 4)]
    return [torch.empty(s, dtype=torch.float64).uniform_(-0.1, 0.1, generator=g) for s in shapes]


def _grad(step: int, i: int, shape) -> torch.Tensor:
    # identical on every rank -> reduce_scatter AVG is exact
    g = torch.Generator().manual_seed(10_000 * step + i)
    return torch.empty(shape, dtype=torch.float64).normal_(0.0, 1.0, generator=g)


def _reference_final(name, cls, lr, mu, wd):
    params = _make_params()
    opt = cls(params, lr=lr, momentum=mu, weight_decay=wd)
    grp = {id(p): g for g in opt.param_groups for p in g["params"]}
    for t in range(T_STEPS):
        for i, p in enumerate(params):
            p.grad = _grad(t, i, tuple(p.shape))
            opt.update_param(p, grp[id(p)])
    # for UDSign the "model" that all_gather broadcasts is W (= the param tensor)
    return [p.detach().clone() for p in params]


def _distributed_final(name, cls, lr, mu, wd):
    params = _make_params()
    opt = cls(params, lr=lr, momentum=mu, weight_decay=wd)
    # locate each param's owning position to assign the right gradient each step
    for t in range(T_STEPS):
        for g in opt.param_groups:
            for p in g["params"]:
                # every rank sets the (identical) grad for every param; step() then
                # reduce_scatters so each owner sees the average (== the grad here)
                idx = _index_of(params, p)
                p.grad = _grad(t, idx, tuple(p.shape))
        opt.step()
    return [p.detach().clone() for p in params]


def _index_of(params, p):
    for i, q in enumerate(params):
        if q is p:
            return i
    raise KeyError


def _worker(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    _install_gloo_shims()

    import signmuon_optimizers as smo
    smo.polar_express = _exact_polar  # stable, exact, fp64 (record #40's LMO slot)

    lr, mu, wd = 0.1, 0.9, 0.0
    failures = []
    for name, cls in smo.OPTIMIZERS.items():
        torch.manual_seed(0)
        ref = _reference_final(name, cls, lr, mu, wd)
        torch.manual_seed(0)
        got = _distributed_final(name, cls, lr, mu, wd)
        max_err = max(float((a - b).abs().max()) for a, b in zip(ref, got))
        ok = max_err < ATOL
        if rank == 0:
            print(f"  {'OK  ' if ok else 'FAIL'} {name:<16} world_size={world_size}  max|Δ|={max_err:.2e}")
        if not ok:
            failures.append(f"{name}: max|Δ|={max_err:.2e} >= {ATOL}")

    dist.barrier()
    dist.destroy_process_group()
    if failures:
        raise AssertionError(f"[rank {rank}] {len(failures)} sharding mismatch(es):\n" + "\n".join(failures))


def main():
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"Distributed sharding test (gloo, world_size={world_size}, {T_STEPS} steps)...\n")
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
    print("\nSharded step() matches the centralized reference for every optimizer. PASS.")


if __name__ == "__main__":
    main()
