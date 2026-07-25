import torch
from contextlib import contextmanager
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(
        G: torch.Tensor, 
        steps: int = 5,
        eps=1e-8,
        ):
    """
    Muon-style Newton–Schulz algorithm to approximate UV^T for a matrix G,
    where G = USV^T is the SVD. 
    Supports tensors with ndim >= 2
    """
    assert G.ndim >= 2, "Muon orthogonalization expects at least 2D tensors."

    device = G.device

    # Newton–Schulz params
    a, b, c = (3.4445, -4.7750, 2.0315)

    # to adopt matrix X
    X = G.to(dtype=torch.bfloat16)
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    # normalization matrix X
    norm = X.norm(dim=(-2, -1), keepdim=True) + eps
    X /= norm

    # steps of Newton–Schulz algorithm
    for _ in range(steps):
        A = X @ X.mT
        A2 = A @ A 
        B = b * A + c * A2
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X.to(device=G.device, dtype=G.dtype)


def muon_orthogonalized_update(
        grad: torch.Tensor, 
        ns_steps: int = 5,
        ):
    """
    Muon-style LMO direction for a gradient tensor.

    - For 4D conv filters: flattens to [out_channels, *], applies NS, reshapes back.
    - For 2D matrices: uses them directly.
    - For 1D / scalars: returns grad unchanged.
    """
    if grad.ndim < 2:
        return grad

    if grad.ndim == 4:
        # Conv filters: [out_channels, in_channels, kh, kw] -> [out_channels, *]
        orig_shape = grad.shape
        G = grad.view(len(grad), -1)
    else:
        G = grad
        orig_shape = None

    orth = zeropower_via_newtonschulz5(G, steps=ns_steps)
    # Same scaling as Muon
    orth = orth * max(1.0, orth.size(-2) / orth.size(-1)) ** 0.5

    if orig_shape is not None:
        orth = orth.view(orig_shape)

    return orth


class SignMuon(Optimizer):
    """
    Centralized SignMuon optimizer.
    LMO <-> Newton–Schulz + sign-comprassed

    For each parameter p:
        m_t = μ m_{t-1} + g_t
        d_t ≈ Muon-LMO(m_t)  (via muon_orthogonalized_update)
        s_t = sign(d_t)
        p_{t+1} = p_t - lr * lambda_mult * s_t
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        norm_weight: bool = False,
        weight_decay=0.0, 
        lambda_mult: float = 1.0,
        ns_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a positive momentum")

        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            norm_weight=norm_weight, weight_decay=weight_decay,
            lambda_mult=lambda_mult, ns_steps=ns_steps
        )
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            norm_weight = group["norm_weight"]
            lambda_mult = group["lambda_mult"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SignMuon does not support sparse gradients")
                
                g = p.grad
                if wd != 0:
                    g = g.add(p.data, alpha=wd)
                state = self.state[p]
                
                # 1) momentum-сглаживание
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                m_t = g.add(buf, alpha=momentum) if nesterov else buf

                # 2) нормализация веса
                if norm_weight:
                    norm = p.data.norm().clamp(min=1e-10)
                    scale = (p.data.numel()**0.5) / norm
                    p.data.mul_(scale)

                # 3) LMO-направление через Newton–Schulz
                d_t = muon_orthogonalized_update(m_t, ns_steps=ns_steps)
                
                # 4) sign‑компрессия Muon‑направления
                s_t = d_t.sign()

                # 5) шаг параметра: x_{t+1} = x_t - lr * lambda_mult * s_t
                p.data.add_(s_t, alpha=-lr * lambda_mult)

        return loss


class MuonSign(Optimizer):
    """
    Centralized MuonSign optimizer (mirror image of SignMuon).
    
    For each parameter p:
        m_t = μ m_{t-1} + g_t                       (momentum)
        s_t = sign(m̃_t)                            (sign compression FIRST)
        d_t ≈ Muon-LMO(s_t)  (via muon_orthogonalized_update)
        p_{t+1} = p_t - lr * lambda_mult * d_t      (full LMO step)
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        norm_weight: bool = False,
        weight_decay=0.0,
        lambda_mult: float = 1.0,
        ns_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a positive momentum")

        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            norm_weight=norm_weight, weight_decay=weight_decay,
            lambda_mult=lambda_mult, ns_steps=ns_steps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            norm_weight = group["norm_weight"]
            lambda_mult = group["lambda_mult"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("MuonSign does not support sparse gradients")

                g = p.grad
                if wd != 0:
                    g = g.add(p.data, alpha=wd)
                state = self.state[p]

                # 1) momentum-сглаживание
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                m_t = g.add(buf, alpha=momentum) if nesterov else buf

                # 2) нормализация веса
                if norm_weight:
                    norm = p.data.norm().clamp(min=1e-10)
                    scale = (p.data.numel()**0.5) / norm
                    p.data.mul_(scale)

                # 3) sign-компрессия моментума (ДО LMO)
                s_t = m_t.sign()

                # 4) LMO-направление через Newton–Schulz от знаковой матрицы
                d_t = muon_orthogonalized_update(s_t, ns_steps=ns_steps)

                # 5) шаг параметра: x_{t+1} = x_t - lr * lambda_mult * d_t
                p.data.add_(d_t, alpha=-lr * lambda_mult)

        return loss


class Muon(Optimizer):
    """
    Centralized Muon optimizer.
    LMO <-> Newton–Schulz (Orthogonalized Momentum)

    For each parameter p:
        m_t = μ m_{t-1} + g_t
        d_t ≈ Muon-LMO(m_t)  (via muon_orthogonalized_update)
        p_{t+1} = p_t - lr * lambda_mult * d_t
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        norm_weight: bool = False,
        weight_decay: float = 0.0,
        lambda_mult: float = 1.0,
        ns_steps: int = 5,
    ):
        if lr < 0.0: 
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0: 
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            nesterov=nesterov,
            norm_weight=norm_weight,
            weight_decay=weight_decay,
            lambda_mult=lambda_mult,
            ns_steps=ns_steps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            norm_weight = group["norm_weight"]
            lambda_mult = group["lambda_mult"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None: 
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SignMuon does not support sparse gradients")
                
                g = p.grad
                if wd != 0:
                    g = g.add(p.data, alpha=wd)
                state = self.state[p]
                
                # 1) momentum-сглаживание
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                m_t = g.add(buf, alpha=momentum) if nesterov else buf

                # 2) нормализация веса
                if norm_weight:
                    norm = p.data.norm().clamp(min=1e-10)
                    scale = (p.data.numel()**0.5) / norm
                    p.data.mul_(scale)

                # 3) LMO-направление через Newton–Schulz
                d_t = muon_orthogonalized_update(m_t, ns_steps=ns_steps)
                
                # 4) шаг параметра
                p.data.add_(d_t, alpha=-lr * lambda_mult)

        return loss


class EF_USignMuon(Optimizer):
    """
    Centralized EF-USignMuon optimizer.

    Single-node reduction of Federated EF-USignMuon (= EF21-Muon with identity
    downlink, Gruntkowska et al. 2025). Sign compression acts on the EF21
    residual (the internal "uplink"); the parameter step is the FULL Muon LMO
    of the reconstructed gradient estimator (NO sign on the step). Applying the
    LMO to the (asymptotically exact) estimator rather than to sign(UV^T) is
    what repairs the SignMuon divergence on the counterexample of Theorem 1.

    For each parameter p:
        m_t   = μ m_{t-1} + g_t                       (momentum)
        Δ_t   = m_t - g_est                           (EF21 residual)
        α_t   = mean(|Δ_t|)
        g_est <- g_est + α_t * sign(Δ_t)              (EF21 estimator, scaled-sign)
        
        d_t   ≈ Muon-LMO(g_est)   (via muon_orthogonalized_update)
       
        p_{t+1} = p_t - lr * lambda_mult * d_t
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        lambda_mult: float = 1.0,
        ns_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            lambda_mult=lambda_mult,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            lambda_mult = group["lambda_mult"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("EF-USignMuon does not support sparse gradients")

                g = p.grad
                if wd != 0:
                    g = g.add(p.data, alpha=wd)
                state = self.state[p]

                # 1) momentum smoothing: m_t = μ m_{t-1} + g_t
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                m_t = g.add(buf, alpha=momentum) if nesterov else buf

                # 2) EF21 gradient estimator (scaled-sign compressor on the residual)
                if "grad_estimator" not in state:
                    state["grad_estimator"] = torch.zeros_like(g)
                g_est = state["grad_estimator"]
                delta = m_t - g_est
                alpha = delta.abs().mean()
                g_est.add_(alpha * torch.sign(delta))   # g_est <- g_est + α * sign(Δ)

                # 3) Muon LMO step on the estimator (NO sign on the step)
                d_t = muon_orthogonalized_update(g_est, ns_steps=ns_steps)

                # 4) parameter update
                p.data.add_(d_t, alpha=-lr * lambda_mult)

        return loss


class EF_UDSignMuon(Optimizer):
    """
    Centralized EF-UDSignMuon optimizer (bidirectional sign compression).

    Faithful single-node reduction of EF21-Muon (Gruntkowska et al. 2025,
    Algorithm 1) with the scaled-sign (1-bit) compressor applied on BOTH
    channels, each carrying its own error-feedback buffer:

        C_up   on the gradient residual (m_t - g_est):   workers -> server
        C_down on the model   increment (X - W):         server  -> workers

        Uplink   EF (gradient):  g_est <- g_est + alpha_up  * sign(m_t - g_est)
        Downlink EF (model):     W     <- W     + alpha_dn * sign(X_new - W)

    For each parameter p:
        g_t      = grad of f at W (= p.data)
        m_t      = μ * m_{t-1} + g_t                      (momentum, heavy-ball)
        Δ_t_up   = m_t - g_est
        alpha_up = mean(|Δ_t_up|)
        g_est    <- g_est + alpha_up  * sign(Δ_t_up)     (uplink, scaled-sign)

        d_t      = Muon-LMO(g_est)
        X_new    = X - lr * lambda_mult * d_t              (exact server step)
        Δ_t_dn   = X_new - W
        alpha_dn = mean(|Δ_t_dn|)
        W        <- W + alpha_dn * sign(Δ_t_dn)         (downlink EF, scaled-sign)
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        lambda_mult: float = 1.0,
        ns_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            lambda_mult=lambda_mult,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            lambda_mult = group["lambda_mult"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("EF-UDSignMuon does not support sparse gradients")

                g = p.grad                       # grad of f at W (= p.data)
                state = self.state[p]

                # exact server model X (init = initial weights = W_0)
                if "exact_model" not in state:
                    state["exact_model"] = p.data.clone()
                X = state["exact_model"]

                # decoupled weight decay on the EXACT model (server side)
                if wd != 0:
                    g = g.add(X, alpha=wd)

                # 1) momentum smoothing: m_t = mu * m_{t-1} + g_t
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                m_t = g.add(buf, alpha=momentum) if nesterov else buf

                # 2) UPLINK error feedback (gradient estimator, scaled-sign)
                if "grad_estimator" not in state:
                    state["grad_estimator"] = torch.zeros_like(g)
                g_est = state["grad_estimator"]
                delta_up = m_t - g_est
                alpha_up = delta_up.abs().mean()
                g_est.add_(alpha_up * torch.sign(delta_up))   # g_est <- g_est + alpha_up * sign(Δ_t_up)

                # 3) Muon LMO direction + exact server step on X
                d_t = muon_orthogonalized_update(g_est, ns_steps=ns_steps)
                X.add_(d_t, alpha=-lr * lambda_mult)          # X <- X - lr * lambda_mult * d_t

                # 4) DOWNLINK error feedback (model increment, scaled-sign)
                delta_dn = X - p.data                          # p.data = W (old broadcast)
                alpha_dn = delta_dn.abs().mean()
                p.data.add_(alpha_dn * torch.sign(delta_dn))   # W <- W + alpha_dn * sign(Δ_t_dn)

        return loss

    @torch.no_grad()
    def restore_exact(self, params=None):
        """Copy the exact server model ``X`` back into ``p.data``.

        During training ``p.data`` holds the compressed broadcast model ``W``;
        call this after training (before evaluation / checkpointing) to expose
        the exact model ``X`` maintained in the ``exact_model`` state buffer.
        """
        groups = self.param_groups if params is None else [{"params": params}]
        for group in groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "exact_model" in st:
                    p.data.copy_(st["exact_model"])

    @contextmanager
    def using_exact(self):
        """Temporarily expose the exact model ``X`` in ``p.data``.

        Saves the broadcast model ``W``, copies ``X`` into ``p.data`` for the
        duration of the ``with`` block, then restores ``W``. Wrap evaluation /
        metric computation of an in-progress ``EF_UDSignMuon`` run so that
        metrics are computed on the exact model while the ``W`` invariant
        (gradient at ``W``) is preserved for subsequent training steps.
        """
        saved = {}
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "exact_model" in st:
                    saved[p] = p.data.clone()
                    p.data.copy_(st["exact_model"])
        try:
            yield
        finally:
            for p, w in saved.items():
                p.data.copy_(w)



class SignSGD(Optimizer):
    """
    Standard SignSGD optimizer.
    Momentum-based Sign Compression

    For each parameter p:
        g_t = g_t + weight_decay * p_t
        m_t = μ m_{t-1} + g_t
        s_t = sign(m_t)
        p_{t+1} = p_t - lr * s_t
    """
    def __init__(
            self, 
            params, 
            lr=1e-3,
            momentum=0.0, 
            nesterov=False, 
            weight_decay=0.0
        ):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None: continue
                
                g = p.grad
                
                # L2 регуляризация (weight decay)
                if weight_decay != 0:
                    g = g.add(p.data, alpha=weight_decay)
                    
                state = self.state[p]
                
                # Momentum
                if momentum != 0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    m_t = g.add(buf, alpha=momentum) if nesterov else buf
                else:
                    m_t = g
                    
                # Sign-компрессия и шаг
                p.data.add_(m_t.sign(), alpha=-lr)

        return loss