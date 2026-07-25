"""
Запуск (на сервере, 1 GPU):
torchrun --standalone --nproc_per_node=1 code/nanogpt/train_nanogpt.py \
  --optimizer signmuon --total_steps 5000 --train_seq_len 2048 --val_seq_len 2048
"""

import os
import sys

try:
    with open(sys.argv[0]) as _f:
        code = _f.read()
except Exception:
    code = ""

import argparse
import glob
import json
import time
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward()  # <--- менять cuda:0
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, flex_attention

# import the optimizers from code/common/optimizers.py (code/ goes on sys.path)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.optimizers import (SignMuon, MuonUSign, MuonSign, Muon,
                               EF21MuonUSign, EF21MuonSign, SignSGD)

# -----------------------------------------------------------------------------
# Communication-volume accounting
# bits/param/step = uplink + downlink
# USign = 1-bit up + full 32-bit down ;  
# UDSign = 1-bit up + 1-bit down ;
# Muon  = full 32-bit both channels.
OPTIMIZER_COMM_BITS = {
    "muon": 64,
    "signsgd": 33,
    "signmuon": 33,
    "muonusign": 33,
    "muonsign": 2,
    "ef21muonusign": 33,
    "ef21muonsign": 2,
}

# -----------------------------------------------------------------------------
# FP8 matmul by @YouJiacheng (used by lm_head when fp8=True)

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=w.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)



# -----------------------------------------------------------------------------
# copied from lesha_nanogpt.py

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5)  # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std  # improved init scale by @YouJiacheng
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, lambdas: Tensor, block_mask: BlockMask):
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v)  # @KoszarskyB & @Grad62304977
        else:
            v = lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()  # https://arxiv.org/abs/2109.08668v2
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor, block_mask: BlockMask):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        x = x + self.mlp(norm(x))
        return x



# -----------------------------------------------------------------------------
def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int,
                 max_seq_len: int, fp8: bool = True):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=fp8,
                                    x_s=(model_dim ** 0.5) / 448, w_s=24 / 448, grad_s=1 / 448)
        self.lm_head.weight.detach().zero_()  # @Grad62304977
        assert num_layers % 2 == 0
        pad = (-num_layers * 5) % dist.get_world_size()
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),                                   # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],   # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],   # SA lambdas
            torch.ones(pad),
        ]))

        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 27.5
        self.scalars.lr_mul = 5.0

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None])

        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)

        n = len(self.blocks) // 2

        for i in range(len(self.blocks)):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1) ** 0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction="sum" if self.training else "mean")
        return loss


# -----------------------------------------------------------------------------
# Data loader (copied from lesha_nanogpt.py; works at world_size==1)

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def find_batch_starts(tokens: Tensor, pos: int, local_batch_size: int, max_batch_span: int):
    boundary_mask = tokens[pos: pos + max_batch_span] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item()
        if end - start >= local_batch_size:
            starts.append(start)
            if len(starts) == dist.get_world_size():
                return starts, end - pos
            start = end
    assert False  # increase max_batch_span if necessary

def distributed_data_generator(filename_pattern: str, batch_size: int, align_to_bos: bool):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert files, f"no data files match {filename_pattern}"
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    max_batch_span = 2 * batch_size if align_to_bos else batch_size
    while True:
        if pos + max_batch_span + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        if align_to_bos:
            batch_starts, batch_span = find_batch_starts(tokens, pos, local_batch_size, max_batch_span)
            start_idx = batch_starts[rank]
        else:
            batch_span = batch_size
            start_idx = pos + rank * local_batch_size
        buf = tokens[start_idx:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_span
        yield inputs, targets


# -----------------------------------------------------------------------------
# Optimizer construction

def _aux_param_groups(params, lr_aux: float, wd_aux: float):
    """Group aux params by their `lr_mul` attribute so plain AdamW replicates the
    per-parameter LR scaling that lesha's DistAdam applied (embed=75, head=27.5, scalars=5.0)."""
    groups = {}
    for p in params:
        lmul = float(getattr(p, "lr_mul", 1.0))
        groups.setdefault(lmul, []).append(p)
    return [{"params": ps, "lr": lr_aux * lmul, "weight_decay": wd_aux} for lmul, ps in groups.items()]


def build_optimizers(model: nn.Module, args):
    """matrix params (2D transformer-block weights, excl. embeddings) -> selected optimizer;
    aux params (embeddings + lm_head + 1D scalars) -> torch.optim.AdamW."""
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]
    aux_params = embed_params + head_params + scalar_params

    name = args.optimizer
    classes = {"signmuon": SignMuon, "muonusign": MuonUSign, "muonsign": MuonSign,
               "muon": Muon, "ef21muonusign": EF21MuonUSign,
               "ef21muonsign": EF21MuonSign, "signsgd": SignSGD}
    if name not in classes:
        raise ValueError(f"Unknown optimizer: {name}")
    main_opt = classes[name](hidden_matrix_params,
                             lr=args.lr, momentum=args.momentum, nesterov=args.nesterov,
                             weight_decay=args.weight_decay,
                             lambda_mult=args.lambda_mult, ns_steps=args.ns_steps)

    aux_groups = _aux_param_groups(aux_params, args.lr_aux, args.weight_decay_aux)
    aux_opt = torch.optim.AdamW(aux_groups, betas=(0.8, 0.95), eps=1e-10) if aux_groups else None
    return main_opt, aux_opt, hidden_matrix_params


# -----------------------------------------------------------------------------
# LR (copied from lesha_nanogpt.py, parameterized)
def get_lr(step: int, total_steps: int, cooldown_frac: float):
    x = step / total_steps
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    w = (1 - x) / cooldown_frac
    return w * 1.0 + (1 - w) * 0.1

@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

def get_window_size_blocks(step: int, total_steps: int):
    x = step / total_steps
    assert 0 <= x <= 1
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)


# -----------------------------------------------------------------------------
# CLI
def get_args():
    p = argparse.ArgumentParser(description="Run optimizers.py optimizers on modded-nanogpt (1 GPU).")
    p.add_argument("--optimizer", type=str, required=True,
                   choices=["signmuon", "muonusign", "muonsign", "muon",
                            "ef21muonusign", "ef21muonsign", "signsgd"])
    # main (matrix) optimizer hyperparameters
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--lambda_mult", type=float, default=1.0)
    p.add_argument("--ns_steps", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    # aux AdamW hyperparameters
    p.add_argument("--lr_aux", type=float, default=0.008)
    p.add_argument("--weight_decay_aux", type=float, default=0.0)
    # schedule / model
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--cooldown_frac", type=float, default=0.45)
    p.add_argument("--train_seq_len", type=int, default=2048)
    p.add_argument("--val_seq_len", type=int, default=2048)
    p.add_argument("--val_every", type=int, default=100)
    p.add_argument("--val_tokens", type=int, default=0,
                   help="Validation tokens; 0 => auto = 16*val_seq_len (a handful of val batches).")
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--momentum_warmup_steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True,
                   help="Use FP8 matmul for lm_head (needs H100/Ada). Disable on other GPUs.")
    # data / output
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--data_dir", type=str,
                   default=os.path.join(here, "modded-nanogpt", "data", "fineweb10B"))
    p.add_argument("--saves_dir", type=str, default=os.path.join(here, "..", "saves_nano"))
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def main():
    args = get_args()
    if args.val_tokens <= 0:
        args.val_tokens = 16 * args.val_seq_len
    assert args.train_seq_len % 128 == 0, "train_seq_len must be a multiple of 128 (FlexAttention block size)"
    assert args.val_seq_len % 128 == 0, "val_seq_len must be a multiple of 128"
    assert args.val_tokens % args.val_seq_len == 0, "val_tokens must be divisible by val_seq_len"

    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 1, (
        "this harness runs single-process (optimizers.py classes do not average gradients "
        "across ranks). Use `torchrun --nproc_per_node=1 ...`."
    )
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    master_process = (rank == 0)

    # ------------------------------------------------------------------
    # log
    logfile = None
    run_name = args.run_name or args.optimizer
    if master_process:
        run_id = str(int(time.time() * 1000))
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)

    def print0(s, console=False):
        if master_process:
            with open(logfile, "a") as f:
                if console:
                    print(s)
                print(s, file=f)

    print0(code)
    print0("=" * 100)
    print0(f"optimizer: {args.optimizer}")
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    print0("=" * 100)

    # ------------------------------------------------------------------ 
    # model
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model: nn.Module = GPT(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768,
                           max_seq_len=max(args.train_seq_len, args.val_seq_len), fp8=args.fp8).cuda()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    # ------------------------------------------------------------------ 
    # optimizers
    main_opt, aux_opt, hidden_matrix_params = build_optimizers(model, args)
    optimizers = [opt for opt in (main_opt, aux_opt) if opt is not None]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # communication-volume accounting 
    numel_matrix = sum(p.numel() for p in hidden_matrix_params)
    comm_bits_per_param = OPTIMIZER_COMM_BITS[args.optimizer]
    bits_per_step = comm_bits_per_param * numel_matrix

    train_files = os.path.join(args.data_dir, "fineweb_train_*.bin")
    val_files = os.path.join(args.data_dir, "fineweb_val_*.bin")

    print0(f"num matrix params: {numel_matrix}  comm bits/param/step: {comm_bits_per_param} "
           f"(bytes/step={bits_per_step/8:.0f})", console=True)
    print0(f"train_files: {train_files}\nval_files: {val_files}", console=True)

    # ------------------------------------------------------------------ 
    # compile + warmup
    if args.compile:
        model = torch.compile(model, dynamic=False)

    warmup_loader = distributed_data_generator(train_files, world_size * args.train_seq_len, align_to_bos=False)
    model.train()
    for _ in range(args.warmup_steps):
        inputs, targets = next(warmup_loader)
        model(inputs, targets, get_window_size_blocks(1, args.total_steps)).backward()
        model.zero_grad(set_to_none=True)
    del warmup_loader

    # ------------------------------------------------------------------ 
    # output dir + metrics
    saves_root = Path(args.saves_dir)
    run_dir = saves_root / run_name
    if master_process:
        run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"

    history = {"step": [], "val_loss": [], "comm_bytes": [], "train_loss": []}
    config = {
        **vars(args),
        "num_matrix_params": numel_matrix,
        "comm_bits_per_param": comm_bits_per_param,
        "bits_per_step": bits_per_step,
        "world_size": world_size,
    }

    def save_metrics():
        if not master_process:
            return
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"config": config, "history": history}, f, indent=2)

    # ------------------------------------------------------------------ 
    # train + val loop
    train_loader = distributed_data_generator(train_files, world_size * args.train_seq_len, align_to_bos=False)
    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_steps = args.total_steps

    for step in range(train_steps + 1):
        last_step = (step == train_steps)

        # ------------------------------ 
        # VALIDATION
        if last_step or (args.val_every > 0 and step % args.val_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * args.val_seq_len
            val_steps = args.val_tokens // val_batch_size
            val_loader = distributed_data_generator(val_files, val_batch_size, align_to_bos=False)

            def compute_val_loss():
                vloss = 0
                with torch.no_grad():
                    for _ in range(val_steps):
                        inputs, targets = next(val_loader)
                        vloss += model(inputs, targets, get_window_size_blocks(step, args.total_steps))
                return vloss / val_steps

            # EF-UD: evaluate on the exact model X, not the compressed broadcast W
            cm = getattr(main_opt, "using_exact", None)
            with (cm() if cm is not None else nullcontext()):
                val_loss = compute_val_loss()
            del val_loader
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            val_loss_f = float(val_loss)
            comm_bytes = bits_per_step * step / 8.0  # `step` train steps completed so far
            history["step"].append(step)
            history["val_loss"].append(val_loss_f)
            history["comm_bytes"].append(comm_bytes)
            print0(f"step:{step}/{train_steps} val_loss:{val_loss_f:.4f} "
                   f"comm_bytes:{comm_bytes:.0f} train_time:{training_time_ms:.0f}ms", console=True)
            save_metrics()

            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step: #for ef-ud
            if hasattr(main_opt, "restore_exact"):
                main_opt.restore_exact()
            break

        # ------------------------------ 
        # TRAINING
        inputs, targets = next(train_loader)
        loss = model(inputs, targets, get_window_size_blocks(step, args.total_steps))
        history["train_loss"].append(float(loss.item()) / inputs.numel())  # per-token train loss
        loss.backward()

        # LR schedule (all groups) + momentum warmup (main optimizer only)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step, args.total_steps, args.cooldown_frac)
        frac = min(step / max(args.momentum_warmup_steps, 1), 1.0)
        for group in main_opt.param_groups:
            group["momentum"] = (1 - frac) * 0.85 + frac * args.momentum

        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step + 1}/{train_steps} train_loss:{history['train_loss'][-1]:.4f} "
               f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms",
               console=True)
        if (step + 1) % args.val_every == 0:
            save_metrics()

    save_metrics()
    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
           f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    if master_process:
        print(f"Saved metrics to: {metrics_path}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
