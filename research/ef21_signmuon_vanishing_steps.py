"""Exact certificate for EF21-SignMuon's nonstationary two-cycle.

Run with Python's standard library:
    python research/ef21_signmuon_vanishing_steps.py
Add --simulate for independent numpy/SVD checks of the actual optimizer.

The certificate uses rational arithmetic throughout. It proves that a unique
algebraic number tau in [251/250, 201/200] defines matrices A, B and S with
    polar(a A + b B) = A + sign(b) B  (a > 0, b != 0),
    <A,S> = 0, sign(A+B) = S, sign(A-B) = -S.
These identities, rather than floating-point experiments, prove the example.
See aaai_article/ef21_signmuon_vanishing_steps.tex for the theorem and proof.
"""

from fractions import Fraction as F
from functools import reduce
from math import gcd, lcm
import argparse


K = [
    [0, -6, -10, 15, 4, -3, -9, 3],
    [6, 0, -3, -7, 0, -12, -4, 0],
    [10, 3, 0, 20, 0, 8, -1, 4],
    [-15, 7, -20, 0, -21, -7, 0, -1],
    [-4, 0, 0, 21, 0, -6, -15, 7],
    [3, 12, -8, 7, 6, 0, -17, 0],
    [9, 4, 1, 0, 15, 17, 0, 12],
    [-3, 0, -4, 1, -7, 0, -12, 0],
]
W = [-1, -8, -9, 0, 0, -12, 7, -8]
J = 5  # zero-based: z(tau) = w + tau e_6
LO, HI = F(251, 250), F(201, 200)


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, col))
             for col in zip(*b)] for row in a]


def matvec(a, v):
    return [sum(x*y for x, y in zip(row, v)) for row in a]


def inverse(a):
    n = len(a)
    aug = [[F(x) for x in row] + [F(i == j) for j in range(n)]
           for i, row in enumerate(a)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if aug[i][j])
        aug[j], aug[pivot] = aug[pivot], aug[j]
        scale = aug[j][j]
        aug[j] = [x/scale for x in aug[j]]
        for i in range(n):
            if i != j:
                scale = aug[i][j]
                aug[i] = [x-scale*y for x, y in zip(aug[i], aug[j])]
    return [row[n:] for row in aug]


def poly_value(p, x):
    """Coefficients are in increasing order."""
    return p[0] + x*(p[1] + x*p[2])


def poly_min(p):
    values = [poly_value(p, LO), poly_value(p, HI)]
    if p[2] > 0:
        vertex = -p[1]/(2*p[2])
        if LO < vertex < HI:
            values.append(poly_value(p, vertex))
    return min(values)


def primitive(p):
    denominator = lcm(*(x.denominator for x in p))
    ints = [int(x*denominator) for x in p]
    divisor = reduce(gcd, ints)
    return [x//divisor for x in ints]


def certificate():
    n = 8
    identity = [[F(i == j) for j in range(n)] for i in range(n)]
    assert all(K[i][j] == -K[j][i] for i in range(n) for j in range(n))
    minus = [[20*identity[i][j]-K[i][j] for j in range(n)] for i in range(n)]
    plus = [[20*identity[i][j]+K[i][j] for j in range(n)] for i in range(n)]
    d = matmul(minus, inverse(plus))
    assert matmul(transpose(d), d) == identity
    s = [[1 if x > 0 else -1 for x in row] for row in d]
    assert min(abs(x) for row in d for x in row) > F(1, 50)

    # q(t) = z(t)^T D^T S z(t); q(tau)=0 is exactly <A,S>=0.
    dw, sw = matvec(d, W), matvec(s, W)
    de, se = [row[J] for row in d], [row[J] for row in s]
    q = [sum(x*y for x, y in zip(dw, sw)),
         sum(x*y+u*v for x, y, u, v in zip(de, sw, dw, se)),
         sum(x*y for x, y in zip(de, se))]
    assert primitive(q) == [-5493727089, 1548933728, 3903331846]
    assert poly_value(q, LO)*poly_value(q, HI) < 0
    assert (q[1]+2*q[2]*LO)*(q[1]+2*q[2]*HI) > 0

    norm = [F(sum(x*x for x in W)), F(2*W[J]), F(1)]
    assert poly_min(norm) > 0
    margins = []
    for i in range(n):
        for j in range(n):
            # N(t) S_ij (D_ij - 2 A_ij(t)) - N(t)/50.
            product = [dw[i]*W[j],
                       de[i]*W[j] + dw[i]*(j == J),
                       de[i]*(j == J)]
            p = [s[i][j]*(d[i][j]*norm[k]-2*product[k])-norm[k]/50
                 for k in range(3)]
            margins.append(poly_min(p))
    assert min(margins) > 0

    h = sum(abs(x) for row in d for x in row)/64
    kappa = 64*h*h
    assert 0 < kappa <= 8
    print("PASS: exact orthogonality, unique algebraic root, and all 128 strict sign inequalities")
    print("q coefficients (constant, linear, quadratic):", primitive(q))
    print("root interval:", LO, HI)
    print("both sign margins exceed 1/50; h = %.12f; kappa = %.12f" % (h, kappa))
    return d, s, q, h, kappa


def simulate(data):
    import numpy as np

    d_exact, s_exact, q, h_exact, kappa_exact = data
    d, s = np.array(d_exact, dtype=float), np.array(s_exact, dtype=float)
    q = np.array(q, dtype=float)
    lo, hi = float(LO), float(HI)
    for _ in range(50):
        mid = (lo+hi)/2
        if poly_value(q, lo)*poly_value(q, mid) <= 0:
            hi = mid
        else:
            lo = mid
    z = np.array(W, dtype=float)
    z[J] += (lo+hi)/2
    a = np.outer(d@z, z)/(z@z)
    b = d-a
    h, kappa = float(h_exact), float(kappa_exact)
    cases = []
    for eta in [0.1, 0.001]:
        for mu in [0.0, 0.9, 0.99]:
            for nesterov in [False, True]:
                cases.append(("constant", eta, mu, nesterov, None))
    # eta_t = r_t+r_{t+1}, r_t = scale/(t+offset)^p.
    for power in [0.5, 1.0]:
        for mu in [0.0, 0.9]:
            offset = 1 if mu == 0 else 20
            for nesterov in [False, True]:
                cases.append(("decreasing", 0.1, mu, nesterov, (power, offset)))

    for kind, scale, mu, nesterov, schedule in cases:
        if schedule is None:
            radius = lambda t: scale/2
        else:
            power, offset = schedule
            radius = lambda t: scale/(t+offset)**power
            assert radius(2) > mu*radius(1)
        center = -kappa*radius(1)
        curvature = 1/(7*(1+center*center))
        offset_a = np.sqrt(1+curvature)
        assert abs((offset_a**2+curvature*center**2)/2-4/7) < 1e-14
        x, momentum, estimator = (np.zeros((8, 8)) for _ in range(3))
        min_grad, max_error = float("inf"), 0.0
        for t in range(1, 2001):
            eta = radius(t)+radius(t+1)
            ax = float(np.sum(a*x))
            grad = (offset_a+ax)*a + curvature*(np.sum(b*x)-center)*b
            min_grad = min(min_grad, float(np.linalg.norm(grad)))
            previous = momentum.copy()
            momentum = mu*momentum+(1-mu)*grad
            signal = (1+mu)*momentum-mu*previous if nesterov else momentum
            u, _, vt = np.linalg.svd(signal, full_matrices=False)
            target = u@vt
            residual = target-estimator
            estimator += np.mean(np.abs(residual))*np.sign(residual)
            x -= eta*estimator
            expected = (-1)**(t-1)*h*s
            max_error = max(max_error, float(np.max(np.abs(estimator-expected))))
        assert min_grad >= 0.999999
        assert max_error < 1e-6
        assert abs(float(np.sum(a*x))) < 1e-6
        print("PASS: %s, scale=%g, mu=%g, nesterov=%s, min|grad|=%.9f, error=%.2g"
              % (kind, scale, mu, nesterov, min_grad, max_error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    result = certificate()
    if args.simulate:
        simulate(result)
