import torch
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
    if norm < eps:
        print("Norm lower than eps")
        return torch.zeros_like(X)
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
        norm_weight: bool = True,
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
        # self.norm_weight = norm_weight
        # self.lambda_mult = lambda_mult
        # self.ns_steps = ns_steps

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
            lambda_mult = group["lambda_mult"]
            ns_steps = group["ns_steps"]
            norm_weight = group["norm_weight"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SignMuon does not support sparse gradients")
                
                g = p.grad 
                if wd != 0:
                    g = g.add(p.data, alpha=wd)               
                state = self.state[p]
                
                # 1) momentum‑сглаживание градиента
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

                # 3) LMO‑направление через Newton–Schulz через ортогонализацию
                #    zeropower_via_newtonschulz5 ожидает 2D тензор -> делаем reshape
                d_t = muon_orthogonalized_update(m_t, ns_steps=ns_steps)
                
                # 4) sign‑компрессия Muon‑направления
                s_t = d_t.sign()

                # 5) шаг параметра: x_{t+1} = x_t - lr * lambda_mult * s_t
                p.data.add_(s_t, alpha=-lr * lambda_mult)

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
        # self.norm_weight = norm_weight
        # self.lambda_mult = lambda_mult
        # self.ns_steps = ns_steps

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
                if p.grad is None: continue
                
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