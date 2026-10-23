# A closed stochastic bound for the experimental fixed-radius update

This note bounds the estimator-error term left explicit in
[the schedule machinery](README.md). It treats EMA and the implemented Nesterov
filter, arbitrary nonnegative deterministic learning rates, both downlinks,
approximate directions, and bounded additive errors in compression updates.
The calculator is [ef21_fixed_momentum.py](../../code/analysis/ef21_fixed_momentum.py).
The Lean components are
[FixedMomentum.lean](../../lean_proofs/EF21Muon/Experiments/FixedMomentum.lean),
[Trajectory.lean](../../lean_proofs/EF21Muon/Experiments/Trajectory.lean), and
[FreshTrajectory.lean](../../lean_proofs/EF21Muon/Experiments/FreshTrajectory.lean),
with the multiple-block extension in
[BlockTrajectory.lean](../../lean_proofs/EF21Muon/Experiments/BlockTrajectory.lean).
The initial derivation below treats one matrix block; the multiple-block
theorem is stated separately near the end. Neither certifies an entire network
trained jointly with AdamW, weight decay or a changing objective.

## Assumptions and the actual round order

All norms without a subscript are Frobenius norms. There are N >= 1 workers.
The global objective is f = mean_j f_j, bounded below by fmin, and is L_f-smooth
in Frobenius norm. Each local gradient g_j is L-Lipschitz in Frobenius norm,
so the global gradient is also L-Lipschitz. These are assumptions, not estimates
extracted from observed losses. Let rho satisfy ||A||_* <= rho ||A|| for all A;
rho = sqrt(min(m,n)) is valid for an m-by-n block.

At round t, fresh replies are a_jt = g_j(W_t) + xi_jt. Conditional on the past,
the noise is centered, independent across workers, and has second moment at
most sigma^2 per worker. Query points depend on the past. Independence of the
queries or of unconditional noise across rounds is not required.

Write mu = 1-beta, with 0 < beta <= 1, and choose theta = 0 for EMA or
theta = beta for Nesterov. Momentum is stored before the round as M_jt:

```text
M_j(t+1) = mu M_jt + beta a_jt
V_jt     = (1-theta) M_j(t+1) + theta a_jt
G_j(t+1) = G_jt + CU(V_jt-G_jt) + zetaU_jt
H_t      = mean_j G_j(t+1)
X_(t+1)  = X_t - eta_t D_t(H_t)
W_(t+1)  = W_t + CP(X_(t+1)-W_t) + zetaP_t.
```

Initialization is X_0=W_0, M_j0=G_j0=0. The theoretical server step is exact.
The direction itself may be approximate: ||D_t|| <= B, ||D_t||_op <= r,
and ||H_t||_* - <H_t,D_t> <= delta_t, where delta_t >= 0 and
the expected learning-rate-weighted delta_t is at most deltaBar.
These are uniform bounds on all possible trajectories; a maximum from one
sampled run is not an a priori bound. eta_t >= 0, sum eta_t = S > 0,
eta_t <= etaMax. No schedule monotonicity is assumed.

For each actual compressor input, require

```text
||R-CU(R)|| <= qU ||R||,  0 <= qU < 1,
||R-CP(R)|| <= qP ||R||,  0 <= qP < 1,
||zetaU_jt|| <= epsU,   ||zetaP_t|| <= epsP.
```

Uniform contraction is one sufficient assumption. Ideal scaled sign has
q=sqrt(1-alpha), with alpha >= 1/(mn); a better alpha needs a valid lower bound
on every possible residual, rather than a sample median. Exact downlink has
qP=epsP=0. Additive errors cover rounding in the communication reconstruction;
rounding in the momentum or server update needs additional terms. The momentum
extension is given below; server rounding is outside the stated theorem.
Standard measurability is required; finite-horizon L2
integrability follows from Lipschitz gradients, bounded server movements,
contractive compressors, and L2 noise by induction.

## Explicit constants and the closed error bound

Define

```text
k        = beta + mu*theta
kappaU   = qU/(1-qU)
P        = (qP*B*etaMax + epsP)/(1-qP)
vEMA     = beta/(2-beta)
vTheta   = k^2 + (1-theta)^2*mu^2*vEMA
G0       = ||grad f(X0)||
H0       = mean_j ||g_j(X0)||
J        = L*B*etaMax + 2*k*L*P
           + sigma*((1-theta)*beta*(1+sqrt(vEMA)) + 2*theta)

Tracking = etaMax/S * ((1-theta)*mu*G0/beta + kappaU*H0)
           + L*P + (1-theta)*mu*L*B*etaMax/beta
           + sigma*sqrt(vTheta/N)
           + kappaU*J + epsU/(1-qU).
```

Then the expected estimator tracking error is explicitly bounded:

```text
sum_t eta_t E ||H_t-grad f(X_t)|| / S <= Tracking.
```

Therefore the previously unresolved nuclear direction-error term satisfies

```text
ErrorBar <= (1+r)*rho*Tracking + deltaBar.
```

With Delta=f(X0)-fmin, the resulting stationarity guarantee is

```text
sum_t eta_t E ||grad f(X_t)||_* / S
  <= Delta/S + (L_f*B^2/2)*(sum_t eta_t^2)/S
     + (1+r)*rho*Tracking + deltaBar.
```

The corresponding bound at W_t adds rho*L*P. Nuclear norms here are first
moments, not squared norms. The guarantee is an upper bound; its residual
terms are not lower bounds on attainable optimization error.

### Multiplicative quality for a numerical LMO

An approximate direction need not leave a constant error floor. If instead
we can certify <H_t,D_t> >= lambda*||H_t||_* - delta_t with a fixed
0 < lambda <= 1, the Lean theorem `radius_direction_gain` proves

```text
<grad f(X_t),D_t> >= lambda*||grad f(X_t)||_*
                    -(lambda+r)*||H_t-grad f(X_t)||_* - delta_t.
```

The closed stationarity bound becomes

```text
[Delta/S + (L_f*B^2/2)*(sum eta_t^2)/S
 + (lambda+r)*rho*Tracking + deltaBar] / lambda.
```

A fixed positive lambda and deltaBar=0 preserve the convergence exponent.
The calculator accepts `oracle_gain=lambda`. A uniform positive gain is a
separate requirement: a few SVD samples cannot certify it for all possible
inputs of the finite-precision Newton-Schulz implementation. The enumerated
calibration below computes its minimum over the entire finite noise support,
which supplies a uniform bound for that small problem only.

### Additive momentum and target-blending roundoff

Suppose the stored momentum update adds a vector with norm at most epsM per
worker, and forming V_jt adds a vector with norm at most epsV. The same proof
holds after the replacements

```text
J        <- J + 2*(1-theta)*epsM + 2*epsV
Tracking <- Tracking + (1-theta)*epsM/beta + epsV
```

where Tracking also uses the modified J. To see this, Step 2's bias cap gains
epsM/beta. In the target increment, the old-momentum error contributes at most
(1-theta)*epsM, the new momentum rounding contributes another
(1-theta)*epsM, and differencing two target-blending errors costs 2*epsV.
The initial target has at most these error bounds as well. No centering of
roundoff is assumed. The calculator exposes `momentum_roundoff` and
`target_roundoff`. In particular, making beta small can amplify systematic
momentum rounding; this term must not be classified as centered gradient noise.

## Proof

### 1. Downlink tracking

Let p_t=||X_t-W_t||. The actual update gives
p_(t+1) <= qP*(p_t+B*eta_t)+epsP. Since p_0=0, induction gives p_t <= P.
Consequently ||W_t-W_(t-1)|| <= B*etaMax+2*P for t>=1.

### 2. Remove the stochastic EMA before bounding momentum lag

Let Z_j0=0 and Z_j(t+1)=mu Z_jt+beta xi_jt. Define
A_jt=M_jt-g_j(X_t)-Z_jt. Direct expansion of the actual update gives

```text
A_j(t+1) = mu A_jt + beta*(g_j(W_t)-g_j(X_t))
           + g_j(X_t)-g_j(X_(t+1)).
```

Hence, pathwise,

```text
||A_jt|| <= mu^t*||g_j(X0)|| + L*P + L*B*etaMax/beta.
```

The same proof for averaged momentum uses G0 in its initial term. Separating
Z before applying the triangle inequality is essential: bounding fresh noise
by E||xi|| at every momentum step would incorrectly leave an order-sigma term
even as beta tends to zero.

### 3. The exact filter variance, including Nesterov

Conditional centering makes Z_jt and xi_jt orthogonal in expectation. Thus

```text
E||Z_j(t+1)||^2 = mu^2 E||Z_jt||^2 + beta^2 E||xi_jt||^2
               <= sigma^2*vEMA.
```

For averaged noise, the same induction has sigma^2/N instead of sigma^2.
The noise component in V_jt is
Y_jt=(1-theta)*mu*Z_jt+k*xi_jt. Orthogonality gives

```text
E||mean_j Y_jt|| <= sigma*sqrt(vTheta/N).
```

Specifically, vTheta=beta/(2-beta) for EMA and
vTheta=beta^2*(2-beta)^2+beta*(1-beta)^4/(2-beta) for Nesterov.
These bounds hold at changing adaptive queries under the conditional noise
assumptions; they are not restricted to a frozen query.

The remaining bias in V averaged across workers is at most

```text
(1-theta)*mu^(t+1)*G0 + L*P + (1-theta)*mu*L*B*etaMax/beta.
```

### 4. Uplink compression tracks the actual Nesterov target

Put V_j,-1=0. For t>=1 the exact increment identity is

```text
V_jt-V_j(t-1) = (1-theta)*beta*(a_jt-M_jt)
               + theta*(a_jt-a_j(t-1)).
```

Use Step 2, E||Z_jt||<=sigma*sqrt(vEMA), E||xi_jt||<=sigma, and the
W-movement bound in Step 1. Averaging over workers gives

```text
mean_j E||V_jt-V_j(t-1)|| <= J + I_t,
I_t = (1-theta)*beta*mu^t*H0 + theta*H0*1_{t=0}.
```

At t=0, V_j0=k*a_j0, and the same bound holds because
J>=k*sigma and I_0=k*H0. The entire initialization impulse has mass
sum_(t>=0) I_t=H0, including beta=1.

For u_t=mean_j E||G_jt-V_j(t-1)||, u_0=0 and

```text
u_(t+1) <= qU*(u_t+J+I_t)+epsU.
```

Let U=kappaU*J+epsU/(1-qU) and b_0=0,
b_(t+1)=qU*(b_t+I_t). Induction gives u_t<=U+b_t. Summing this recursion
and dropping a nonnegative terminal term yields
sum_(t<T) b_(t+1) <= kappaU*H0. Therefore, for any schedule bounded by etaMax,

```text
sum_(t<T) eta_t*u_(t+1)/S <= U + etaMax*kappaU*H0/S.
```

This argument requires no unjustified independence between compression
residuals and their achieved contraction ratios. It uses the stated uniform qU.

### 5. Close the direction error and telescope descent

The norm of the averaged uplink error is at most the average norm. Combine
Steps 3 and 4, and use sum_(t<T) eta_t*mu^(t+1) <= etaMax*mu/beta.
This proves Tracking above. The already proved approximate-direction inequality
and Frobenius smoothness give

```text
E f(X_(t+1)) <= E f(X_t) - eta_t E||grad f(X_t)||_*
  + eta_t*((1+r)*rho*E||H_t-grad f(X_t)||+E delta_t)
  + (L_f*B^2/2)*eta_t^2.
```

Telescoping proves the stationarity bound. Query transfer uses
||grad f(W_t)-grad f(X_t)||_* <= rho*L*P.

## Rates and interpretation

For fixed beta and a constant or cosine schedule with a fixed positive peak,
the objective-gap and initialization terms decay as 1/T. The remaining terms
include learning-rate, filtered-noise, compression and numerical residuals.
With the peak scaled as T^(-1/2), the bound is O(T^(-1/2)) plus the remaining
fixed-beta noise and numerical terms (all other problem constants fixed).

With exact arithmetic and exact polar directions, set beta proportional to
T^(-1/2) and the peak proportional to T^(-3/4). Constant and cosine profiles
then both give O(T^(-1/4)) weighted nuclear first-moment stationarity, for
either downlink and either momentum filter, under ordinary Frobenius smoothness.
For cosine the guarantee is learning-rate weighted; it does not automatically
control the unweighted average or the last iterate. Nonvanishing oracle or
communication errors add the explicitly displayed terms.

This does not establish all-regime linear speedup: only the filtered averaged
noise has 1/sqrt(N); local uplink compression still has a sigma term in J.
For small alpha, q/(1-q) is approximately 2/alpha. The additive downlink
roundoff floor is epsP/(1-qP), also approximately 2*epsP/alphaP.

This explains why a large matrix with poor downlink contraction can have a
weak guarantee even when momentum effectively filters the averaged noise.
Using measured per-input multipliers gives a retrospective trajectory bound;
it cannot replace the uniform lower bound in this a priori stochastic result.

## Lean scope

`FixedMomentum.lean` checks the vector identities, exact filter variance,
initialization impulse and scalar closure. `Trajectory.lean` now derives the
primitive recurrences and finite-horizon L2 integrability from the actual
N-worker optimizer updates and composes them with objective descent.
`expected_stationarity` bounds the expectation of the weighted nuclear-gradient
score along the complete random trajectory on one probability space, with the
same closed constants displayed above.

`FreshTrajectory.lean`, theorem `fresh_stationarity`, starts from a filtration:
noise in round t is revealed in F_(t+1), has conditional mean zero given F_t,
and has per-worker second moment at most sigma^2. Same-round conditional worker
cross moments vanish; conditional independence is sufficient but not necessary.
Cross-time orthogonality is proved using conditional expectation, rather than
assuming independence of adaptive replies. Causal reconstruction errors give
adapted server, broadcast and worker-memory states. No tracking recurrence,
objective-drift recurrence, trajectory integrability or stationarity bound is
a premise.

The composed theorem covers theta=0 and theta=beta, arbitrary nonnegative
deterministic schedules, both downlinks, epsU/epsP communication errors, and
uniform multiplicative/additive direction quality. It uses exact momentum,
target-blending and server arithmetic. The optional epsM/epsV extension above
remains analytical. This theorem treats one matrix block under ordinary
smoothness. The extension below uses the paper's layerwise assumptions;
neither gives a certificate from a sampled alpha statistic. The formalization
does not provide a new rate exponent.

## Multiple blocks under the earlier paper assumptions

`BlockTrajectory.fresh_stationarity` in
[BlockTrajectory.lean](../../lean_proofs/EF21Muon/Experiments/BlockTrajectory.lean)
proves the full simultaneous N-worker trajectory for
`X=(X_1,...,X_p)`, with differently sized matrix blocks. Its objective and local
gradient assumptions are exactly the arbitrary-pair layerwise assumptions used
by the earlier ordinary-smooth theorems and the supplied Gruntkowska paper:

```text
||grad_i f(X)-grad_i f(Y)||_* <= L_i ||X_i-Y_i||_op,
||grad_i f_j(X)-grad_i f_j(Y)||_* <= L_ij ||X_i-Y_i||_op,
for every pair of full models X,Y and every block i and worker j.
```

`Problem.ofDistributed` directly converts the earlier distributed problem to
this problem structure. Setting X_i=Y_i proves that the population gradient
in block i is independent of the other blocks. The code derives that fact and
uses it to identify each block of the actual simultaneous full-query run with
the existing single-block trajectory. It does not add separability as an extra
assumption. The largest worker constant `Lbar_i=max_j L_ij` supplies the block's
Frobenius Lipschitz constant; there is no extra N factor from summing constants.

All blocks share one filtration. Conditional centering and zero cross moments
are required between workers within each block; noise between different blocks
may be correlated. Each block may have its own beta_i, theta_i, compressors,
communication error bounds and arbitrary nonnegative deterministic eta_it
bounded by etaMax_i. EMA uses theta_i=0; Nesterov uses theta_i=beta_i.
All memories start at zero and X_0=W_0. Momentum, target blending and server
arithmetic remain exact.

For each block use the previous tracking formulas with its own parameters,
L=Lbar_i and sigma=sigma_i. Write

```text
S_i = sum_(t<T) eta_it,           Q_i = sum_(t<T) eta_it^2,
G_i = ||grad_i f(X_0)||_F,        H_i = mean_j ||grad_i f_j(X_0)||_F,
B_i = etaMax_i*((1-theta_i)*(1-beta_i)*G_i/beta_i
                  + qUi/(1-qUi)*H_i) + S_i*(A_i+U_i).
```

Here A_i and U_i are `noiseCap` and `compressionCap`, respectively, from the
single-block proof. Thus `B_i` is the already closed `trackingBudget`, not an
assumed estimator-error bound. It includes downlink tracking, filtered noise
with the exact `sqrt(filterGain(beta_i,theta_i)/N)` factor, local compression
noise and communication roundoff.

Let the direction satisfy
`||D_it(H)||_op <= r_i` and
`gain_i*||H||_*-delta_it <= <H,D_it(H)>`, with gain_i>0, and let rho_i bound
nuclear norm by Frobenius norm. Put

```text
K_i = (gain_i+r_i)*rho_i,
Sbar = (sum_i gain_i*S_i)/p > 0,
Delta = f(X_0)-fmin.
```

The normalized theorem is

```text
E[ sum_(t<T) sum_i eta_it*gain_i*||grad_i f(X_t)||_* / Sbar ]
 <= [ Delta + sum_i (K_i*B_i + sum_(t<T) eta_it*delta_it
                              + (L_i*r_i^2/2)*Q_i) ] / Sbar.
```

The global objective is telescoped once: Delta is not multiplied by p. The
curvature uses the spectral radius r_i, as supplied by layerwise smoothness.
Mean-layer normalization agrees with the earlier paper's weighted-sum
convention. In particular, equal schedules eta_it=eta_t and unit gains give

```text
sum_(t<T) eta_t * E[sum_i ||grad_i f(X_t)||_*] / S
 <= [Delta + sum_i (K_i*B_i + sum_(t<T) eta_t*delta_it
                             + (L_i*r_i^2/2)*Q)] / S,
S=sum_(t<T) eta_t,  Q=sum_(t<T) eta_t^2.
```

The theorem includes adaptation of every server/broadcast block and every
worker memory. Tracking recurrences, objective drift, integrability and the
stationarity estimate are proved from the updates. This extends the scope to
multiple blocks under the inherited assumptions; it does not prove a sharper
stochastic exponent or a result for general interacting neural-network layers.

## Numerical verification and reproduction

[fixed_momentum_calibration.json](fixed_momentum_calibration.json) enumerates
every Rademacher noise history in eight small cases: both momentum filters,
both downlinks, and constant/cosine schedules. The shared Newton-Schulz LMO
is used on a known 2-by-2 quadratic; compatible ties are fixed to +1. Its
direction norm, additive defect, and minimum relative pairing are bounded over
all histories, rather than one sampled path. Each case checks the expected
tracking error, stationarity, noise variance, broadcast gap, and target-increment
bound. These are numerical checks of the analytical proof, not a floating-point
formal verification. From `code/`:

```sh
python -m analysis.ef21_fixed_calibrate --steps 5 --output ../research/experiments/fixed_momentum_calibration.json
python -m unittest tests.test_fixed_momentum -v
```

The tests also enumerate independent noise for two heterogeneous workers at
adaptive broadcast queries. These are CPU calibrations; no new NN training
run is represented here.
