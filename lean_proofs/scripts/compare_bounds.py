"""Evaluate the finite-sampling Lean bounds; this does not run an optimizer."""

import argparse
import json
import math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--smoothness", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=10000)
    parser.add_argument("--gap", type=float, default=1.0)
    parser.add_argument("--initial-gradient", type=float, default=1.0,
                        help="Frobenius norm of the initial gradient")
    parser.add_argument("--sigma", type=float, default=0.0)
    args = parser.parse_args()
    if min(args.rows, args.cols, args.rounds) <= 0:
        parser.error("rows, cols, and rounds must be positive")
    values = (args.beta, args.smoothness, args.gap, args.initial_gradient, args.sigma)
    if not all(math.isfinite(v) for v in values):
        parser.error("all real parameters must be finite")
    if not 0 < args.beta <= 1 or min(values[1:]) < 0:
        parser.error("require 0 < beta <= 1 and nonnegative remaining parameters")
    if args.smoothness == 0:
        parser.error("use positive smoothness to obtain a finite reciprocal threshold")

    rank = min(args.rows, args.cols)
    alpha = 1.0 / (args.rows * args.cols)
    beta = args.beta
    scale = rank * args.smoothness**2
    exact = scale * (12 / beta**2 + 36 * (beta**2 + 4) / alpha**2)
    # Theorem 19: identity downlink, alpha_P = 1; spectral compression class.
    paper_usign = exact + scale * (
        24 * (beta + 2) + 144 * beta**2 * (2 * beta + 5) / alpha**2)
    # Remark 23: scaled-sign downlink, alpha_P = alpha; Frobenius class.
    paper_sign = exact + scale * rank * (
        24 * (beta + 2) / alpha**2
        + 144 * beta**2 * (2 * beta + 5) / alpha**4)
    weight = 6 * rank / beta + 72 * rank * beta / alpha**2
    noise = 6 * rank * beta * (
        1 + (1 - alpha) * beta / alpha + 12 * beta**2 / alpha**2)

    def evaluate(zeta):
        gamma = 1 / (2 * args.smoothness + 2 * math.sqrt(zeta))
        return {
            "zeta": zeta,
            "gamma_max": gamma,
            "average_squared_gradient_upper_bound": (
                4 * args.gap / (gamma * args.rounds)
                + 4 * weight * args.initial_gradient**2 / args.rounds
                + 4 * noise * args.sigma**2),
        }

    results = {
        "scope": "one worker, one layer, finite independent sampling; bound calculator",
        "parameters": vars(args),
        "muon_usign_paper": evaluate(paper_usign),
        "muon_usign_exact_downlink": evaluate(exact),
        "muon_sign_paper_remark23": evaluate(paper_sign),
    }
    results["muon_usign_step_threshold_ratio"] = (
        results["muon_usign_exact_downlink"]["gamma_max"]
        / results["muon_usign_paper"]["gamma_max"])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
