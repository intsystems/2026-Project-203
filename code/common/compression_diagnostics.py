"""Opt-in, per-layer EF21-P measurements. No RNG draws or optimizer changes.

These are measurements of a finite trajectory, not population convergence
certificates. Tensor operations and host synchronization occur only on selected
steps. Use every=1 to reconstruct the full tracking envelope.
"""

import csv
import math
import re
from pathlib import Path


FIELDS = (
    "schema", "layer", "step", "sample_every", "eta", "beta", "nesterov", "dimension", "finite",
    "alpha_input", "effective_support", "max_energy_share", "signed_mass_ratio",
    "gap_before_sq", "server_step_sq", "gap_step_inner", "residual_sq",
    "compressor_input_sq", "post_gap_sq", "observed_q", "identity_defect",
    "broadcast_norm_sq", "relative_gap",
    "uplink_error_sq", "uplink_target_sq", "direction_sq", "estimator_pairing",
    "estimator_nuclear", "direction_operator", "oracle_pairing_defect",
)


def residual_metrics(residual, compressor_input, post_gap, gap_before, server_step,
                     broadcast):
    """All norms are Frobenius; alpha is per compressor block, never a median.

    residual = X_new-W_old, post_gap = X_new-W_new. Keeping the actual
    post-gap accounts for quantization and rounding in W += C(R).
    """
    import torch

    tensors = [x.detach().double() for x in
               (residual, compressor_input, post_gap, gap_before, server_step, broadcast)]
    if any(x.shape != tensors[0].shape for x in tensors):
        raise ValueError("diagnostic tensors must have the same shape")
    d = tensors[0].numel()
    if d == 0:
        raise ValueError("empty compressor block")
    if not all(bool(torch.isfinite(x).all()) for x in tensors):
        return {"dimension": d, "finite": False}
    r, inp, post, gap, move, w = tensors
    r2, i2, p2, g2, u2, w2 = [float(x.square().sum()) for x in tensors]
    l1 = float(inp.abs().sum())
    alpha = l1*l1/(d*i2) if i2 else None
    return dict(
        dimension=d, finite=True, alpha_input=alpha,
        effective_support=d*alpha if alpha is not None else None,
        max_energy_share=float(inp.square().max())/i2 if i2 else None,
        signed_mass_ratio=abs(float(inp.sum()))/l1 if l1 else None,
        gap_before_sq=g2, server_step_sq=u2,
        gap_step_inner=float((gap*move).sum()), residual_sq=r2,
        compressor_input_sq=i2, post_gap_sq=p2,
        observed_q=math.sqrt(p2/r2) if r2 else (0.0 if p2 == 0 else None),
        identity_defect=p2/i2-(1-alpha) if i2 else None,
        broadcast_norm_sq=w2, relative_gap=math.sqrt(p2/w2) if w2 else None,
    )


class CompressionRecorder:
    """CSV sidecar for independent compressor blocks, with explicit round IDs.

    begin/end surround the *entire* server step, including weight decay. Each
    layer has its own counter, so skipped gradients do not fabricate samples.
    The analyzer checks continuity of both IDs and gap energies.
    """

    def __init__(self, path, every=1, audit_oracle=False, layers=""):
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError("every must be a positive integer")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.audit_oracle = audit_oracle
        self.layers = re.compile(layers)
        self.counts = {}
        self.pending = {}
        self.context = {}
        # Refuse to mix independent runs or append a second CSV header.
        with self.path.open("x", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def measure_direction(self, layer, target, estimator, direction, step=None):
        """Track the uplink lag; optionally audit the LMO using exact SVD.

        SVD is diagnostic only and never replaces the optimizer's direction.
        It is expensive on large blocks. The population gradient is not observed.
        """
        if not self.layers.search(layer):
            return
        step = self.counts.get(layer, 0)+1 if step is None else step
        if (step-1) % self.every:
            return
        import torch
        estimator, direction = [v.detach().double() for v in (estimator, direction)]
        target = target.detach().double() if target is not None else None
        tensors = (estimator, direction) if target is None else (target, estimator, direction)
        if not all(bool(torch.isfinite(v).all()) for v in tensors):
            return
        measured = dict(direction_sq=float(direction.square().sum()),
                        estimator_pairing=float((estimator*direction).sum()))
        if target is not None:
            measured.update(uplink_error_sq=float((estimator-target).square().sum()),
                            uplink_target_sq=float(target.square().sum()))
        if self.audit_oracle:
            def matrix(tensor):
                return tensor.reshape(tensor.shape[0], -1) if tensor.ndim >= 2 else tensor.reshape(1, -1)
            measured["estimator_nuclear"] = float(torch.linalg.svdvals(matrix(estimator)).sum())
            measured["direction_operator"] = float(torch.linalg.svdvals(matrix(direction)).max())
            measured["oracle_pairing_defect"] = measured["estimator_nuclear"]-measured["estimator_pairing"]
        self.context[layer] = measured

    def begin(self, layer, exact, broadcast, eta, beta, nesterov=False, step=None):
        if not self.layers.search(layer):
            return
        count = self.counts.get(layer, 0)+1
        self.counts[layer] = count
        step = count if step is None else step
        if (step-1) % self.every:
            return
        self.pending[layer] = (
            exact.detach().double().clone(), broadcast.detach().double().clone(),
            dict(schema="ef21p-v1", layer=layer, step=step, sample_every=self.every, eta=float(eta),
                 beta=float(beta), nesterov=bool(nesterov)),
        )

    def end(self, layer, exact, broadcast, compressor_input):
        saved = self.pending.pop(layer, None)
        if saved is None:
            return
        old_x, old_w, row = saved
        x, w = exact.detach().double(), broadcast.detach().double()
        row.update(residual_metrics(x-old_w, compressor_input, x-w,
                                    old_x-old_w, x-old_x, w))
        row.update(self.context.pop(layer, {}))
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
