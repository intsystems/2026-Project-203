"""Explicit stochastic fixed-radius EF21 error budgets.

Inputs are mathematical upper bounds, not estimates inferred from sampled
training medians. The assumptions are stated in fixed_momentum_bound's docstring.
"""

import math


def fixed_momentum_bound(rates, *, beta, nesterov, workers, sigma, smoothness,
                         local_lipschitz, direction_frobenius, direction_operator,
                         nuclear_frobenius, alpha_up, alpha_down, initial_gap,
                         initial_global_gradient, initial_local_gradient,
                         oracle_defect=0.0, oracle_gain=1.0,
                         uplink_roundoff=0.0, downlink_roundoff=0.0,
                         momentum_roundoff=0.0, target_roundoff=0.0):
    """Bound expected weighted nuclear-gradient stationarity at X and W.

    Initialization: X=W, all momenta/estimators zero. Fresh centered noise with
    per-worker Frobenius variance <= sigma**2 and independent workers is required.
    Each local gradient is local_lipschitz Lipschitz in Frobenius norm. Ideal
    compressors have achieved alpha >= the supplied deterministic lower bounds
    on every possible input. Roundoff bounds are absolute Frobenius errors per
    compressor update. The oracle pairing is >= oracle_gain*||H||_*-oracle_defect.
    The server step itself is assumed exact.
    """
    values = [float(x) for x in rates]
    if not values or any(not math.isfinite(x) or x < 0 for x in values) or not math.fsum(values) > 0:
        raise ValueError("a finite nonnegative schedule with positive mass is required")
    if not math.isfinite(beta) or not 0 < beta <= 1:
        raise ValueError("beta must be in (0,1]")
    if not math.isfinite(oracle_gain) or not 0 < oracle_gain <= 1:
        raise ValueError("oracle_gain must be in (0,1]")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    for name, value in (("alpha_up", alpha_up), ("alpha_down", alpha_down)):
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(name+" must be in (0,1]")
    bounds = (sigma, smoothness, local_lipschitz, direction_frobenius, direction_operator,
              nuclear_frobenius, initial_gap, initial_global_gradient,
              initial_local_gradient, oracle_defect, uplink_roundoff, downlink_roundoff,
              momentum_roundoff, target_roundoff)
    if any(not math.isfinite(x) or x < 0 for x in bounds):
        raise ValueError("all norm, error, gap and Lipschitz bounds must be finite and nonnegative")
    mass = math.fsum(values)
    square_mass = math.fsum(x*x for x in values)
    peak = max(values)
    mu = 1-beta
    theta = beta if nesterov else 0.0
    k = beta+mu*theta

    def contraction(alpha):
        q = math.sqrt(1-alpha)
        # Stable at alpha << 1: 1-q = alpha/(1+q).
        inverse_memory = (1+q)/alpha
        return q, inverse_memory

    qu, memory_u = contraction(alpha_up)
    qp, memory_p = contraction(alpha_down)
    kappa_u = qu*memory_u
    gap = memory_p*(qp*direction_frobenius*peak+downlink_roundoff)
    ema_gain = beta/(2-beta)
    gain = k*k+(1-theta)**2*mu*mu*ema_gain
    increment = (local_lipschitz*direction_frobenius*peak+2*k*local_lipschitz*gap
                 + sigma*((1-theta)*beta*(1+math.sqrt(ema_gain))+2*theta)
                 + 2*(1-theta)*momentum_roundoff+2*target_roundoff)
    terms = dict(
        initialization=peak/mass*((1-theta)*mu*initial_global_gradient/beta
                                 + kappa_u*initial_local_gradient),
        broadcast_lag=local_lipschitz*gap,
        momentum_lag=(1-theta)*mu/beta*local_lipschitz*direction_frobenius*peak,
        filtered_noise=sigma/math.sqrt(workers)*math.sqrt(gain),
        uplink_tracking=kappa_u*increment,
        uplink_roundoff=memory_u*uplink_roundoff,
        momentum_roundoff=(1-theta)*momentum_roundoff/beta,
        target_roundoff=target_roundoff,
    )
    tracking = math.fsum(terms.values())
    geometry = (oracle_gain+direction_operator)*nuclear_frobenius
    initial_term = initial_gap/mass
    curvature = smoothness*direction_frobenius**2/2*square_mass/mass
    server = (initial_term+curvature+geometry*tracking+oracle_defect)/oracle_gain
    transfer = nuclear_frobenius*local_lipschitz*gap
    return dict(
        steps=len(values), beta=beta, theta=theta, nesterov=bool(nesterov), workers=workers,
        sum_eta=mass, peak_eta=peak, q_up=qu, q_down=qp,
        kappa_up=kappa_u, downlink_gap_bound=gap, filter_variance_gain=gain,
        oracle_gain=oracle_gain,
        target_increment_forcing=increment, frobenius_error_terms=terms,
        frobenius_tracking_bound=tracking, nuclear_direction_error_bound=geometry*tracking+oracle_defect,
        initial_gap_term=initial_term, curvature_term=curvature,
        server_stationarity_bound=server, query_transfer_term=transfer,
        broadcast_stationarity_bound=server+transfer,
        scope="expected first-moment weighted stationarity under supplied uniform bounds; not a certificate inferred from an NN trace",
    )
