"""Two CNN2/CIFAR-10 label-skew experiments for the rebuttal.

From code/:
    python -m federated.heterogeneous --experiment transfer --download
    python -m federated.heterogeneous --experiment tuned --tune-part priority
    python -m federated.heterogeneous --experiment tuned --tune-part remaining

Transfer runs all ten article methods with their homogeneous learning rates.
Tuning is split into priority and remaining methods, and selects LR only.
Momentum stays 0.9 throughout. All three commands share one cumulative archive;
completed jobs and identical finals are reused automatically. Planning and
reporting need only the Python standard library.
Use --refine-methods within a tuning part to continue cached boundary searches
until the validation winner has tested learning rates on both sides.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

from common.paths import results_root, scrub_text
from common.lr_grid import extend_grid, round_grid


# Validation-selected rates in results/federated_overnight/state.json, phase lr;
# also recorded in results/federated_export/table_federated.csv. All use mu=0.9.
HOMOGENEOUS_LRS = {
    "signmuon": 0.1, "ef21muonusign": 0.01, "ef21muonsign": 0.05,
    "ef21signmuon": 0.02, "muon": 0.1, "muonusign": 0.05, "muonsign": 0.02,
    "signsgd": 0.01, "sgd": 0.05, "adam": 0.001,
}
PRIORITY_METHODS = ["signmuon", "ef21muonusign", "ef21muonsign", "ef21signmuon", "muon"]
REMAINING_METHODS = ["muonusign", "muonsign", "signsgd", "sgd", "adam"]
DEFAULT_METHODS = PRIORITY_METHODS + REMAINING_METHODS
HOMOGENEOUS_MOMENTUM = 0.9
# Nearest lattice points to federated.tune.anchor_for(..., "unit-gain", CNN2).
# These reproduce the homogeneous initial grids, including its sign-family
# anchor transport. They are search centers, distinct from the selected LRs.
HOMOGENEOUS_SEARCH_CENTERS = {
    "signmuon": 0.02, "ef21muonusign": 0.02, "ef21muonsign": 0.05,
    "ef21signmuon": 0.02, "muon": 0.05, "muonusign": 0.05,
    "muonsign": 0.02, "signsgd": 0.02, "sgd": 0.02, "adam": 0.001,
}


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=["transfer", "tuned", "both"], default="transfer")
    p.add_argument("--methods", nargs="+", choices=sorted(HOMOGENEOUS_LRS),
                   default=DEFAULT_METHODS)
    p.add_argument("--tune-part", choices=["priority", "remaining"], default="priority",
                   help="Methods to tune; transfer always runs the entire --methods list")
    p.add_argument("--alpha", type=float, default=0.1, help="Dirichlet concentration")
    p.add_argument("--rounds", type=int, default=2000, help="Same horizon for tuning and finals")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--tune-seeds", nargs="+", type=int, default=[0])
    p.add_argument("--lr-points", type=int, default=5,
                   help="Initial 1-2-5 lattice points, matching homogeneous tuning")
    p.add_argument("--lr-extend-rounds", type=int, default=3)
    p.add_argument("--lr-extend-points", type=int, default=2)
    p.add_argument("--refine-methods", nargs="+", choices=sorted(HOMOGENEOUS_LRS), default=[],
                   help="Continue cached searches in this tuning part until the validation winner is interior")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default="./data_federated")
    p.add_argument("--download", action="store_true")
    p.add_argument("--loader", choices=["auto", "gpu", "torch"], default="auto")
    p.add_argument("--dry-run", action="store_true", help="Print plan without training or writing")
    p.add_argument("--report-only", action="store_true", help="Rebuild tables/archive without training")
    p.add_argument("--retry-failed", action="store_true", help="Retry recorded failures on this invocation")
    args = p.parse_args(argv)
    if not math.isfinite(args.alpha) or args.alpha <= 0 or args.rounds < 1:
        p.error("alpha must be finite and positive; rounds must be positive")
    if args.lr_points < 1 or args.lr_extend_rounds < 0 or args.lr_extend_points < 1:
        p.error("LR points and extension points must be positive; extension rounds nonnegative")
    if any(x < 0 for x in args.seeds + args.tune_seeds):
        p.error("seeds must be nonnegative")
    for name in ("methods", "seeds", "tune_seeds", "refine_methods"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            p.error(f"duplicate entries in --{name.replace('_', '-')}")
    if args.refine_methods:
        if args.experiment != "tuned":
            p.error("--refine-methods requires --experiment tuned")
        group = PRIORITY_METHODS if args.tune_part == "priority" else REMAINING_METHODS
        if any(m not in args.methods or m not in group for m in args.refine_methods):
            p.error("--refine-methods must belong to --methods and the selected --tune-part")
    if args.experiment != "transfer" and not args.report_only and not tuning_methods(args):
        p.error(f"no requested methods belong to tuning part {args.tune_part!r}")
    return args


def tuning_methods(args):
    group = PRIORITY_METHODS if args.tune_part == "priority" else REMAINING_METHODS
    return [m for m in args.methods if m in group
            and (not args.refine_methods or m in args.refine_methods)]


def tuning_part(method):
    return "priority" if method in PRIORITY_METHODS else "remaining"


def protocol(args):
    # Experiment and tuning part are scheduling choices, absent from identity:
    # all three commands share one cache, report, and cumulative archive.
    return {
        "version": 3, "dataset": "cifar10", "model": "cnn2", "rounds": args.rounds,
        "n_parties": 11, "n_steps": 3, "batch_size": 64,
        "partition": "noniid-labeldir", "beta": args.alpha,
        "partition_seed": "run seed", "val_seed": 12345,
        "lr_aux": 0.001, "lr_scaling": "unit-gain", "weight_decay": 0.0,
        "nesterov": False, "cosine_schedule": True, "freeze_bn_stats": True,
        "ns_steps": 5, "lmo_dtype": "bfloat16", "mv_ties": "random",
        "uplink_zeros": "random", "last_k": 5, "eval_points": 20,
        "methods": args.methods, "seeds": args.seeds, "tune_seeds": args.tune_seeds,
        "momentum": HOMOGENEOUS_MOMENTUM,
        "homogeneous_lrs": {m: HOMOGENEOUS_LRS[m] for m in args.methods},
        "homogeneous_momentum": HOMOGENEOUS_MOMENTUM,
    }


def protocol_id(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def legacy_protocol(spec):
    """The original default protocol, for resuming completed transfer runs."""
    return {**spec, "version": 2, "lr_factors": [0.1, 0.3, 1, 3, 10]}


def protocol_directory(spec):
    base = results_root() / "federated_heterogeneous"
    current = base / protocol_id(spec)
    old = base / protocol_id(legacy_protocol(spec))
    return old if not (current / "state.json").exists() and (old / "state.json").exists() else current


def search_protocol(args):
    return {"grid": "1-2-5", "initial_points": args.lr_points,
            "extension_rounds": args.lr_extend_rounds, "extension_points": args.lr_extend_points,
            "centers": {m: HOMOGENEOUS_SEARCH_CENTERS[m] for m in args.methods}}


def candidates(args, method, grid=None):
    rates = round_grid(HOMOGENEOUS_SEARCH_CENTERS[method], args.lr_points) if grid is None else grid
    return [(lr, HOMOGENEOUS_MOMENTUM) for lr in rates]


def job_key(method, lr, momentum, seed, split):
    return f"{method}_lr{float(lr)!r}_mu{float(momentum)!r}_{split}_s{seed}"


def select_best(args, jobs, method, grid=None):
    """Use validation only; a candidate needs every requested tuning seed."""
    if any(job_key(method, lr, mu, s, "tune") not in jobs
           for lr, mu in candidates(args, method, grid) for s in args.tune_seeds):
        return None  # An interrupted grid does not yet define a tuned setting.
    ranked = []
    for lr, mu in candidates(args, method, grid):
        rows = [jobs.get(job_key(method, lr, mu, s, "tune"), {}) for s in args.tune_seeds]
        if all(r.get("status") == "ok" and "val_acc" in r for r in rows):
            score = statistics.mean(r["val_acc"] for r in rows)
            # Exact ties prefer the homogeneous setting, then deterministic order.
            tie = (abs(math.log(lr / HOMOGENEOUS_LRS[method])), lr)
            ranked.append((-score, tie, lr, mu))
    if not ranked:
        return None
    _, _, lr, mu = min(ranked)
    return lr, mu


@contextmanager
def isolated_results(root):
    """Both run_one and its child honor this existing results-root override."""
    previous = os.environ.get("SIGNMUON_RESULTS")
    os.environ["SIGNMUON_RESULTS"] = str(root.resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SIGNMUON_RESULTS", None)
        else:
            os.environ["SIGNMUON_RESULTS"] = previous


def save_state(root, state):
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(root / "state.json")


def run_job(args, root, state, method, lr, mu, seed, split, runner):
    key = job_key(method, lr, mu, seed, split)
    old = state["jobs"].get(key)
    if old and ((old["status"] == "ok" and (root / old["metrics"]).is_file())
                or (old["status"] != "ok" and not args.retry_failed)):
        return old
    tag = f"hetero_{root.name}_{key}"
    # Match tune.run_one's canonical tag; do not resolve its scrubbed path, which
    # intentionally omits the machine-specific prefix of a custom results root.
    canonical = f"{tag}_e{args.rounds}_{split[0]}s{seed}"
    relative = Path("raw/federated") / f"{'tune' if split == 'tune' else 'final'}_{canonical}" / f"seed{seed}/metrics.json"
    row = {"method": method, "lr": lr, "momentum": mu, "seed": seed,
           "partition_seed": seed, "split": split, "status": "failed",
           "log": f"raw/federated_tuning_logs/{canonical}.log"}
    launch = argparse.Namespace(
        dataset="cifar10", model="cnn2", n_parties=11, n_steps=3, batch_size=64,
        momentum=mu, weight_decay=0.0, partition="noniid-labeldir", beta=args.alpha,
        val_seed=12345, last_k=5, target_acc=80.0, eval_points=20,
        seed=seed, device=args.device, data=args.data, loader=args.loader,
        mv_ties="random", uplink_zeros="random", download=args.download,
    )
    print(f"[{split}] {method}: lr={lr:g}, mu={mu:g}, seed={seed}", flush=True)
    started = time.perf_counter()
    with isolated_results(root / "raw"):
        result = runner(launch, lr=lr, lr_aux=0.001, lr_scaling="unit-gain",
                        method=method, rounds=args.rounds, tag=tag, split=split,
                        seed=seed, extra=["--partition-seed", str(seed)])
    row["wall_seconds"] = time.perf_counter() - started
    if result is not None:
        try:
            metrics = json.loads((root / relative).read_text(encoding="utf-8"))
            config, history = metrics["config"], metrics["history"]
            if config["rounds"] != args.rounds or history["steps"][-1] != args.rounds:
                raise ValueError("run did not reach the requested horizon")
            metric = "val_acc" if split == "tune" else "test_acc"
            values = [float(x) for x in history[metric] if x is not None]
            if not values or not all(math.isfinite(x) for x in values):
                raise ValueError(f"missing or nonfinite {metric}")
            row.update(status="ok", metrics=relative.as_posix(),
                       **{metric: statistics.mean(values[-5:])})
        except (OSError, KeyError, ValueError, IndexError) as exc:
            row["error"] = scrub_text(str(exc))
    state["jobs"][key] = row
    save_state(root, state)
    return row


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tuned_setting(args, state, method):
    entry = state.get("searches", {}).get(method, {})
    return (select_best(args, state["jobs"], method, entry["grid"])
            if entry.get("complete") else None)


def tune_method(args, root, state, method, runner, outcomes):
    """Use the homogeneous search, or continue a requested boundary refinement."""
    entry = state["searches"].setdefault(method, {
        "grid": [lr for lr, _ in candidates(args, method)], "extensions": 0, "complete": False})
    if method in args.refine_methods and "refinement" not in entry:
        previous = select_best(args, state["jobs"], method, entry["grid"])
        entry["refinement"] = {
            "stop_rule": "interior_validation_winner", "initial_grid": list(entry["grid"]),
            "starting_extensions": entry["extensions"],
            "previous_lr": previous[0] if previous else None,
            "complete": False,
        }
    refining = "refinement" in entry
    if refining and entry.get("boundary"):
        entry["complete"] = False
    if args.retry_failed:
        entry["complete"] = False
    save_state(root, state)
    while not entry["complete"]:
        for lr, mu in candidates(args, method, entry["grid"]):
            for seed in args.tune_seeds:
                outcomes.append(run_job(args, root, state, method, lr, mu, seed, "tune", runner))
                report(args, root, state)
        if refining and any(state["jobs"][job_key(method, lr, mu, seed, "tune")]["status"] != "ok"
                            for lr, mu in candidates(args, method, entry["grid"])
                            for seed in args.tune_seeds):
            raise RuntimeError(f"{method}: refinement has failed validation jobs; rerun with --retry-failed")
        best = select_best(args, state["jobs"], method, entry["grid"])
        boundary = best and len(entry["grid"]) > 1 and best[0] in (min(entry["grid"]), max(entry["grid"]))
        if not boundary or (not refining and entry["extensions"] >= args.lr_extend_rounds):
            entry["complete"] = True
            entry["boundary"] = bool(boundary)
        else:
            entry["grid"] = extend_grid(entry["grid"], low=best[0] == min(entry["grid"]),
                                        points=args.lr_extend_points)
            entry["extensions"] += 1
            limit = "until an interior winner" if refining else f"limit {args.lr_extend_rounds}"
            print(f"  Extending {method} to {entry['grid']} ({entry['extensions']} extensions; {limit})")
        if refining:
            entry["refinement"]["complete"] = entry["complete"] and not entry.get("boundary", False)
        save_state(root, state)
    if refining:
        entry["refinement"]["complete"] = not entry.get("boundary", False)
        save_state(root, state)


def report(args, root, state, archive=False):
    """Keep transfer and validation-selected finals in separate table rows."""
    rows, notes = [], []
    for experiment in state["experiments"]:
        for method in args.methods:
            chosen = ((HOMOGENEOUS_LRS[method], HOMOGENEOUS_MOMENTUM)
                      if experiment == "transfer" else tuned_setting(args, state, method))
            lr, mu = chosen if chosen else (None, HOMOGENEOUS_MOMENTUM)
            finals = ([state["jobs"].get(job_key(method, lr, mu, s, "full"), {}) for s in args.seeds]
                      if chosen else [])
            values = [r["test_acc"] for r in finals if r.get("status") == "ok"]
            row = {"experiment": experiment, "method": method,
                   "tuning_part": tuning_part(method) if experiment == "tuned" else "",
                   "lr": lr, "momentum": mu,
                   "seeds": len(values), "expected_seeds": len(args.seeds),
                   "acc_mean": statistics.mean(values) if values else None,
                   "acc_std": statistics.stdev(values) if len(values) > 1 else None,
                   "complete": len(values) == len(args.seeds)}
            rows.append(row)
            if experiment == "tuned" and chosen:
                lrs = sorted(state["searches"][method]["grid"])
                if len(lrs) > 1 and lr in (lrs[0], lrs[-1]):
                    notes.append(f"{method}: selected LR {lr:g} remains at the boundary after "
                                 f"{state['searches'][method]['extensions']} extensions.")
    jobs = list(state["jobs"].values())
    failures = sum(r["status"] != "ok" for r in jobs)
    if failures:
        notes.append(f"{failures} failed jobs are recorded in state.json; --retry-failed retries them.")
    columns = ["experiment", "tuning_part", "method", "lr", "momentum", "seeds", "expected_seeds",
               "acc_mean", "acc_std", "complete"]
    write_csv(root / "table.csv", rows, columns)
    write_csv(root / "tuning.csv", [r for r in jobs if r["split"] == "tune"],
              ["method", "lr", "momentum", "seed", "partition_seed", "status", "val_acc", "metrics"])
    write_csv(root / "runs.csv", jobs,
              ["method", "lr", "momentum", "seed", "partition_seed", "split", "status",
               "val_acc", "test_acc", "wall_seconds", "metrics", "log", "error"])
    lines = ["# CNN2 heterogeneous experiments", "",
             f"CIFAR-10; Dirichlet concentration {args.alpha:g}; 11 clients; full participation; "
             f"{args.rounds} rounds; 3 x 64 gradient accumulation at the broadcast model; cosine schedule.", "",
             "Transfer runs the full article family using homogeneous validation-selected LRs. "
             "Tuned selects LR only by the last-five validation accuracy, averaged over tuning seeds. "
             "Momentum stays 0.9 in both experiments. "
             "The validation split holds out 5,000 images; finals retrain on all 50,000. "
             "Auxiliary AdamW LR stays 0.001. Client objectives have equal weight.", "",
             f"Final seeds: {args.seeds}; tuning seeds: {args.tune_seeds}. The run seed also seeds "
             "the client partition; methods share initialization and partition at each seed. "
             "Reported accuracy is the mean +/- sample SD over seeds of each run's last-five test mean.", "",
             "| Experiment | Tuning part | Method | LR | Momentum | Seeds | Test accuracy (%) |",
             "| :-- | :-- | :-- | --: | --: | --: | --: |"]
    for r in rows:
        acc = "pending" if r["acc_mean"] is None else f"{r['acc_mean']:.2f}"
        if r["acc_std"] is not None:
            acc += f" +/- {r['acc_std']:.2f}"
        if not r["complete"]:
            acc += " (incomplete)"
        lr = "pending" if r["lr"] is None else f"{r['lr']:g}"
        lines.append(f"| {r['experiment']} | {r['tuning_part']} | {r['method']} | {lr} | {r['momentum']:g} | "
                     f"{r['seeds']}/{r['expected_seeds']} | {acc} |")
    lines += ["", "## Search", "",
              f"Homogeneous 1-2-5 lattice: {args.lr_points} initial points, at most "
              f"{args.lr_extend_rounds} boundary extensions of {args.lr_extend_points} points; fixed momentum 0.9. "
              "Initial search centers match the homogeneous CNN2 experiment. "
              "Base searches receive the same search rule, seeds, and horizon. Each method's full grid "
              "finishes before validation selection; test accuracy never enters selection. "
              "Identical final configurations are reused across experiments.", "",
              f"Priority tuning: {', '.join(m for m in args.methods if m in PRIORITY_METHODS)}.", "",
              f"Remaining tuning: {', '.join(m for m in args.methods if m in REMAINING_METHODS)}.", "",
              f"Tuning parts started: {', '.join(state['tuning_parts']) or 'none'}. "
              "Pending rows identify methods whose tuning or final seeds are unfinished.", ""]
    for method, entry in state.get("searches", {}).items():
        lines.append(f"- {method}: grid {entry['grid']}; extensions {entry['extensions']}; "
                     f"search {'complete' if entry['complete'] else 'pending'}.")
        if "refinement" in entry:
            refinement = entry["refinement"]
            extra = entry["extensions"] - refinement["starting_extensions"]
            lines.append(f"  Additional refinement: continue past the base extension budget until an interior "
                         f"validation winner; {extra} extra extensions; "
                         f"{'complete' if refinement['complete'] else 'pending'}. "
                         f"Previous selected LR: {refinement['previous_lr']}.")
    lines += [f"- {note}" for note in notes]
    lines += ["", "Files: table.csv (final results), tuning.csv (validation trials), runs.csv, state.json, "
              "raw/federated/.../metrics.json (full configs and curves), and per-job logs. "
              "The archive excludes weights.", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    if archive:
        with zipfile.ZipFile(root / "rebuttal_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("SUMMARY.md", "table.csv", "tuning.csv", "runs.csv", "state.json"):
                z.write(root / name, name)
            for row in jobs:
                for field in ("metrics", "log"):
                    if row.get(field) and (root / row[field]).is_file():
                        z.write(root / row[field], row[field])
    return rows


def run(args, runner=None):
    spec = protocol(args)
    root = protocol_directory(spec)
    path = root / "state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "protocol": spec, "jobs": {}, "experiments": [], "tuning_parts": []}
    if state["protocol"] == legacy_protocol(spec):
        state["previous_protocol"] = state["protocol"]
        state["protocol"] = spec
        print("Resuming the existing transfer cache; completed runs are preserved.")
    if state["protocol"] != spec:
        raise ValueError("recorded protocol differs; refusing to mix experiments")
    state.setdefault("searches", {})
    search = search_protocol(args)
    if state.get("tuning_protocol", search) != search and state["searches"] and not args.report_only:
        raise ValueError("recorded LR search settings differ; keep the original search settings")
    if args.refine_methods:
        missing = [m for m in args.refine_methods if m not in state["searches"]]
        if missing:
            raise ValueError(f"no cached search for {', '.join(missing)}; keep the original results tree "
                             "and --methods list when refining")
    experiments = ["transfer", "tuned"] if args.experiment == "both" else [args.experiment]
    n_final = len(args.methods) * len(args.seeds)
    selected_tuning = tuning_methods(args)
    n_tune = len(selected_tuning) * args.lr_points * len(args.tune_seeds)
    n_max = len(selected_tuning) * (args.lr_points + args.lr_extend_points * args.lr_extend_rounds) * len(args.tune_seeds)
    print(f"Dirichlet alpha={args.alpha:g}; methods={', '.join(args.methods)}")
    if args.refine_methods:
        print(f"Refine: {', '.join(selected_tuning)}; continue cached grids until an interior validation winner. "
              f"Up to {len(selected_tuning) * len(args.seeds)} finals at the selected LR; fixed momentum 0.9.")
    else:
        print(f"Transfer: {n_final} finals (all requested methods, one experiment).")
        print(f"Tuned/{args.tune_part}: {n_tune} initial validation trials ({n_max} maximum with extensions) + up to "
              f"{len(selected_tuning) * len(args.seeds)} finals; fixed momentum 0.9.")
    print(f"Selected experiment(s): {', '.join(experiments)}; {args.rounds} rounds per job.")
    print(f"Results: {root}")
    if args.dry_run:
        if "transfer" in experiments:
            for method in args.methods:
                print(f"  transfer/{method}: lr={HOMOGENEOUS_LRS[method]:g}, mu=0.9")
        if "tuned" in experiments:
            for method in selected_tuning:
                if args.refine_methods:
                    entry = state["searches"][method]
                    print(f"  refine/{method}: cached LRs={entry['grid']}; "
                          f"{entry['extensions']} completed extensions; stop at an interior winner, mu=0.9")
                    best = select_best(args, state["jobs"], method, entry["grid"])
                    if best and best[0] in (min(entry["grid"]), max(entry["grid"])):
                        expanded = extend_grid(entry["grid"], low=best[0] == min(entry["grid"]),
                                               points=args.lr_extend_points)
                        print(f"    Next new LRs: {sorted(set(expanded) - set(entry['grid']))}")
                else:
                    print(f"  tuned/{args.tune_part}/{method}: "
                          f"LRs={sorted({lr for lr, _ in candidates(args, method)})}, mu=0.9")
        return root
    if not args.report_only:
        state["tuning_protocol"] = search
    if not args.report_only:
        for experiment in experiments:
            if experiment not in state["experiments"]:
                state["experiments"].append(experiment)
        if "tuned" in experiments and args.tune_part not in state["tuning_parts"]:
            state["tuning_parts"].append(args.tune_part)
    save_state(root, state)
    outcomes = []
    try:
        if not args.report_only:
            if runner is None:
                from federated.tune import run_one
                runner = run_one
            for experiment in experiments:
                methods = args.methods if experiment == "transfer" else selected_tuning
                if experiment == "tuned":
                    for method in methods:
                        tune_method(args, root, state, method, runner, outcomes)
                chosen_settings = {}
                for method in methods:
                    chosen = ((HOMOGENEOUS_LRS[method], HOMOGENEOUS_MOMENTUM)
                              if experiment == "transfer" else tuned_setting(args, state, method))
                    if chosen is None:
                        raise RuntimeError(f"no valid validation candidate for {method}; inspect state.json")
                    chosen_settings[method] = chosen
                # Seed-major finals produce a complete one-seed table first.
                for seed in args.seeds:
                    for method, chosen in chosen_settings.items():
                        outcomes.append(run_job(args, root, state, method, *chosen, seed, "full", runner))
                        report(args, root, state)
    finally:
        report(args, root, state, archive=True)
        print(f"Summary: {root / 'SUMMARY.md'}\nDownload: {root / 'rebuttal_results.zip'}")
    if any(r["status"] != "ok" for r in outcomes):
        raise RuntimeError("some jobs failed; inspect state.json and rerun with --retry-failed")
    return root


if __name__ == "__main__":
    run(get_args())
