"""Exact scalar checks for the separate Nesterov proof (requires SymPy).

These verify the coefficient calculations and squared-norm certificates.
They are not a formalization of the stochastic convergence theorems.
Run: python research/nesterov/check_algebra.py
"""

import sympy as s


def identical(name, left, right):
    difference = s.factor(left - right)
    if difference != 0:
        raise AssertionError((name, difference))
    print("PASS:", name)


beta, alpha = s.symbols("beta alpha", positive=True)
a = 1 - beta
u, v, g, error = s.symbols("u v g error", real=True)

# Each scalar identity is the coefficient identity of a Hilbert-space
# squared-norm identity. The extra coefficients are nonnegative on
# 0 < beta <= 1 and 0 < alpha <= 1.
identical(
    "EMA weighted Young certificate",
    a * u**2 + a**2 / beta * v**2 - a**2 * (u + v)**2,
    a * beta * (u - a / beta * v)**2,
)
identical(
    "contracted-sum certificate",
    (1 - alpha / 2) * u**2 + 2 / alpha * v**2
    - (1 - alpha) * (u + v)**2,
    alpha / 2 * (u - 2 * (1 - alpha) / alpha * v)**2
    + (3 - alpha) * v**2,
)
identical(
    "sharp pairing certificate at the worst permitted gradient norm",
    g**2 - g * error
    - ((g + error)**2 / 4 + g**2 / 2 - s.Rational(5, 2) * error**2),
    (g - 3 * error)**2 / 4,
)

old, sample = s.symbols("old sample", real=True)
momentum = a * old + beta * sample
target = a * momentum + beta * sample
identical("Nesterov target identity", target, (1 + a) * momentum - a * old)
identical(
    "conditional local variance difference",
    (beta * (1 + a))**2 - a**2 * beta**2,
    beta**2 * (1 + 2 * a),
)
identical("noise EMA geometric denominator", 1 - a**2, beta * (2 - beta))

# Check the combined server-error coefficients, independently expanding
# the three component sums in the proof.
F0, drift, movement, L, delta, K, T, sigma2 = s.symbols(
    "F0 drift movement L delta K T sigma2", nonnegative=True
)
Fsum = F0 / beta + drift / beta**2 + beta * T * sigma2
Hsum = 160 * beta / alpha**2 * F0 + 160 / alpha**2 * drift \
    + 120 * beta**2 / alpha**2 * T * sigma2
combined = 3 * Hsum + 3 * Fsum + 9 * beta**2 * T * sigma2 \
    + 3 * L**2 * delta * movement
A = 3 / beta + 480 * beta / alpha**2
B = (3 / beta**2 + 480 / alpha**2) * K + 3 * L**2 * delta
C = 3 * beta + (9 + 360 / alpha**2) * beta**2
identical(
    "smooth server-error coefficients",
    combined.subs(drift, K * movement),
    A * F0 + B * movement + C * T * sigma2,
)
Z = (3 + 480 / alpha**2) * K + 3 * L**2 * delta
identical(
    "smooth tuning envelope remainder",
    Z - beta**2 * B,
    (1 - beta**2) * (480 / alpha**2 * K + 3 * L**2 * delta),
)

# Generalized-smooth descent: recover the coefficients from the tracking
# estimate rather than trusting the displayed grouped expression.
radius, rho, kappa = s.symbols("radius rho kappa", nonnegative=True)
global_coeff, local_coeff = s.symbols("global_coeff local_coeff")
U = radius**2 * (s.Rational(1, 2) + 2 * rho / beta)
V = 12 * kappa * radius**2 * rho
identical(
    "generalized-smooth coefficient grouping",
    radius**2 / 2 * global_coeff
    + 2 * radius * rho
    * (radius / beta * global_coeff + 6 * kappa * radius * local_coeff),
    U * global_coeff + V * local_coeff,
)

# Write h = T^(-1/4), so beta = h^2, radius = c h^3, T = h^(-4).
h, c = s.symbols("h c", positive=True)
schedule = {beta: h**2, radius: c * h**3}
identical(
    "T times generalized global coefficient",
    U.subs(schedule) / h**4,
    c**2 * (h**2 / 2 + 2 * rho),
)
identical(
    "T times generalized local coefficient",
    V.subs(schedule) / h**4,
    12 * kappa * c**2 * rho * h**2,
)
identical(
    "generalized compression-noise power",
    (radius * beta).subs(schedule) / h**4,
    c * h,
)
identical(
    "generalized filtered-noise power",
    (radius * s.sqrt(beta)).subs(schedule) / h**4,
    c,
)
identical(
    "generalized stationarity normalization",
    radius.subs(schedule) / h**4,
    c / h,
)

print("Exact algebra checks passed. See the TeX note for the full proof.")
