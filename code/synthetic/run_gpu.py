"""One command to produce every synthetic-benchmark number, on a GPU box.

    python3 -m synthetic.run_gpu --quick             # ~5 min, verify the pipeline
    python3 -m synthetic.run_gpu                     # the real thing, ~5 h
    python3 -m synthetic.run_gpu --stages floor horizon
    python3 -m synthetic.run_gpu --summarize-only    # rebuild SUMMARY.md, no runs

Everything before the ``grid`` stage takes about 1h45 all told and is the new
material; ``grid`` is the ~3 h Table 1 re-run. ``--stages`` splits them if you
would rather not hold one job open for five hours.

Run from the ``code/`` directory. **Do the ``--quick`` pass first** -- it runs
every stage at 64x64 with a coarse grid, so a mistake costs five minutes instead
of five hours.

Each stage is a separate ``synthetic.benchmark`` subprocess, so one failure does
not take the others down, and every stage is skipped if its output already
exists (``--force`` to redo). Everything lands under ``results/synthetic/``:

    results/synthetic/
        SUMMARY.md              <- every table in one file; this is the one to read
        MANIFEST.json           <- commit, GPU, torch version, argv and timing per stage
        logs/<stage>.log        <- full console output of that stage
        <method>/<mode>.json    <- the machine-readable results

To hand the results back for analysis, open ``SUMMARY.md`` (and any
``logs/*.log`` that looks wrong). ``results/`` is not tracked by git, so the
files stay local until you share them.

Stage order is cheapest-first, so the interesting measurements arrive long
before the expensive table re-run finishes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from common.utils import results_root

CODE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Stage:
    """One ``synthetic.benchmark`` invocation."""

    name: str
    mode: str
    why: str
    args: List[str] = field(default_factory=list)
    quick_args: List[str] = field(default_factory=list)
    estimate: str = "?"

    @property
    def out_key(self) -> str:
        """Basename this stage writes, matching ``benchmark.output_slug``.

        Two stages can share a ``mode`` and still be distinct measurements
        (``grid`` vs ``grid-paper``), so the skip-if-done check and the summary
        must key on the file, not on the mode.
        """
        if "--grid-preset" in self.args:
            return f"grid-{self.args[self.args.index('--grid-preset') + 1]}"
        return self.mode

    def argv(self, device: str, out: Path, quick: bool,
             size: Optional[tuple] = None) -> List[str]:
        base = [sys.executable, "-m", "synthetic.benchmark",
                "--mode", self.mode, "--device", device, "--out", str(out)]
        argv = base + self.args + (self.quick_args if quick else [])
        if size is not None:
            # Appended last so it beats any --m/--n baked into the stage's own
            # args (argparse keeps the final occurrence).
            argv += ["--m", str(size[0]), "--n", str(size[1])]
        return argv


# Cheapest first. Every stage below `grid` runs at the module's 100x100 default;
# `grid` and `final` keep the paper's 500x500.
STAGES: List[Stage] = [
    Stage(
        "stability", "stability",
        "Largest stable eta per method. SGD is the built-in control: its "
        "eta_max must land on the textbook 2/L, and if it does not, nothing "
        "else in this run is trustworthy.",
        estimate="~5 min",
        quick_args=["--m", "64", "--n", "64", "--stability-iters", "150",
                    "--problem-seeds", "1337"],
    ),
    Stage(
        "alignment", "alignment",
        "Distribution of rho_t = <grad F, d_t>/(||grad F|| ||d_t||) along the "
        "tuned trajectory -- the quantity the descent lemma needs positive and "
        "the divergence theorems drive negative. The one measurement here that "
        "is about the methods rather than the tuning protocol.",
        estimate="~20 min",
        quick_args=["--m", "64", "--n", "64", "--align-iters", "300",
                    "--problem-seeds", "1337", "--momentum-grid", "0.0,0.9",
                    "--lr-grid", "signmuon=1e-4:1e-2:x2", "muonsign=1e-4:1e-2:x2",
                    "signsgd=1e-4:1e-2:x2", "ef21signmuon=1e-4:1e-2:x2",
                    "muon=1e-3:1e-1:x2", "muonusign=1e-3:1e-1:x2",
                    "ef21muonusign=1e-3:1e-1:x2", "ef21muonsign=1e-3:1e-1:x2",
                    "sgd=1e-2:1e0:x2", "adam=1e-2:1e0:x2"],
    ),
    Stage(
        "floor", "floor",
        "The plateau F_inf(eta), ||grad F||_inf(eta) and their exponents. This "
        "is what Table 1's iteration count is actually ranking; the descent "
        "lemma predicts a gradient floor linear in eta.",
        # 1.5 decades is plenty to fit a slope, and the budget scales as 1/eta,
        # so widening the grid costs superlinearly for no extra information.
        args=["--problem-seeds", "1337", "--floor-iters", "3000",
              "--floor-max-iters", "60000",
              "--lr-grid", "signmuon=6e-5:2e-3:x4", "muonsign=6e-5:2e-3:x4",
              "signsgd=6e-5:2e-3:x4", "ef21signmuon=6e-5:2e-3:x4",
              "muon=6e-4:2e-2:x4", "muonusign=6e-4:2e-2:x4",
              "ef21muonusign=6e-4:2e-2:x4", "ef21muonsign=6e-4:2e-2:x4",
              "sgd=6e-3:2e-1:x4", "adam=6e-4:2e-2:x4"],
        estimate="~25 min",
        quick_args=["--m", "64", "--n", "64", "--floor-iters", "1200",
                    "--floor-max-iters", "20000", "--problem-seeds", "1337",
                    "--methods", "signmuon", "signsgd", "muonsign", "muon",
                    "--lr-grid", "signmuon=2e-4:2e-3:x3", "muonsign=2e-4:2e-3:x3",
                    "signsgd=2e-4:2e-3:x3", "muon=2e-3:2e-2:x3"],
    ),
    Stage(
        "horizon", "horizon",
        "Tunes (eta, momentum, schedule) separately at each budget T and fits "
        "err ~ T^-p, eta* ~ T^-q. Says which regime the problem is in: p = 1/2 "
        "is the nonconvex bound the theorems prove, p = 1 is strongly convex. "
        "Also lets each method pick its own schedule rather than imposing one.",
        # The lr grid stays dense (6 per decade) because q is fitted from
        # eta*(T), and a coarse grid quantizes it into noise. The budget list and
        # the instance count are what get trimmed instead.
        args=["--problem-seeds", "1337", "--momentum-grid", "0.0,0.9",
              "--budgets", "125", "250", "500", "1000", "2000"],
        estimate="~45 min",
        quick_args=["--m", "64", "--n", "64", "--budgets", "125", "250", "500",
                    "--problem-seeds", "1337", "--momentum-grid", "0.0,0.9",
                    "--methods", "signmuon", "signsgd", "muon", "sgd",
                    "--lr-grid", "signmuon=1e-4:1e-2:x3", "signsgd=1e-4:1e-2:x3",
                    "muon=1e-3:1e-1:x3", "sgd=1e-2:1e1:x3"],
    ),
    Stage(
        "kappa", "kappa",
        "The tuned comparison at a controlled condition number, swept over "
        "decades. Conditioning is the only knob that governs quadratic "
        "dynamics and the paper currently reports one uncontrolled draw.",
        args=["--problem-seeds", "1337", "--momentum-grid", "0.0,0.9",
              "--max-iters", "3000"],
        estimate="~30 min",
        quick_args=["--m", "40", "--n", "40", "--max-iters", "1500",
                    "--kappas", "1e2", "1e3", "1e4", "--problem-seeds", "1337",
                    "--methods", "signmuon", "signsgd", "muon",
                    "--momentum-grid", "0.0,0.9",
                    "--lr-grid", "signmuon=1e-4:1e-2:x3", "signsgd=1e-4:1e-2:x3",
                    "muon=1e-3:1e-1:x3"],
    ),
    Stage(
        "grid", "grid",
        "THE TABLE 1 / TABLE 3 RE-RUN, at the paper's 500x500. Log-spaced "
        "grids 3-4 decades wide, because Table 3's one-decade linear grids left "
        "SGD censored at its own boundary. Any row still flagged [BOUNDARY] "
        "needs its grid widened again before it can be published. Roughly half "
        "the configurations never reach the target and so run the full 5000 "
        "iterations, which is what dominates the cost; split by --methods if "
        "you would rather not hold a single job open that long.",
        estimate="~3 h",
        quick_args=["--m", "64", "--n", "64", "--max-iters", "1500",
                    "--methods", "signmuon", "signsgd", "muon", "sgd",
                    "--momentum-grid", "0.0,0.9",
                    "--lr-grid", "signmuon=1e-4:1e-2:x3", "signsgd=1e-4:1e-2:x3",
                    "muon=1e-3:1e-1:x3", "sgd=1e-2:1e1:x3"],
    ),
    Stage(
        "final", "final",
        "Re-runs the tuned optima with --save-histories, for the Figure 4 loss "
        "and gradient-norm curves. Reads the optima from the grid stage above, "
        "not from the paper's printed table.",
        args=["--save-histories"],
        estimate="~5 min",
        quick_args=["--m", "64", "--n", "64", "--max-iters", "1500",
                    "--methods", "signmuon", "signsgd", "muon", "sgd"],
    ),
    Stage(
        "grid-paper", "grid",
        "OPTIONAL. Table 3's grids verbatim, to confirm the published numbers "
        "and show which rows were boundary-limited. Only worth the time if a "
        "reviewer asks how the old table was produced; --stages grid-paper.",
        args=["--grid-preset", "paper"],
        estimate="~2 h",
        quick_args=["--m", "64", "--n", "64", "--max-iters", "1500",
                    "--methods", "signmuon", "signsgd", "sgd"],
    ),
]

#: Not in the default run: it re-does work only to document a known defect.
OPTIONAL = {"grid-paper"}

BY_NAME = {s.name: s for s in STAGES}


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def environment(device: str) -> Dict:
    """Everything a reader needs to know the run was what it claims to be."""
    # One source of truth for the machine record, so the paper's hardware table
    # can be built from any results tree (`python3 -m common.hardware --scan`).
    from common.hardware import describe
    info: Dict = dict(describe(device))
    info["hardware"] = dict(info)

    for key, cmd in (("git_commit", ["git", "rev-parse", "HEAD"]),
                     ("git_branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                     ("git_dirty", ["git", "status", "--porcelain"])):
        try:
            out = subprocess.run(cmd, cwd=CODE_ROOT, capture_output=True,
                                 text=True, timeout=30)
            info[key] = out.stdout.strip() if key != "git_dirty" else bool(out.stdout.strip())
        except Exception:                                   # noqa: BLE001
            info[key] = None
    return info


def run_stage(stage: Stage, device: str, out: Path, log_dir: Path,
              quick: bool, size: Optional[tuple] = None) -> Dict:
    """Run one stage, teeing its output to console and to a log file."""
    argv = stage.argv(device, out, quick, size)
    log_path = log_dir / f"{stage.name}.log"
    print(f"\n{'=' * 78}\n[{stage.name}]  {stage.estimate}\n{stage.why}\n"
          f"$ {' '.join(argv[1:])}\n{'=' * 78}", flush=True)

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(argv)}\n\n")
        log.flush()
        proc = subprocess.Popen(argv, cwd=CODE_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        code = proc.wait()
    elapsed = time.time() - start

    status = "ok" if code == 0 else f"FAILED (exit {code})"
    print(f"[{stage.name}] {status} in {elapsed / 60:.1f} min -> {log_path}",
          flush=True)
    return {"stage": stage.name, "mode": stage.mode, "argv": argv,
            "exit_code": code, "seconds": round(elapsed, 1),
            "log": str(log_path)}


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def _load(out: Path, mode: str) -> List[Dict]:
    """Every ``<method>/<mode>.json`` under ``out``, in METHOD_CLASSES order."""
    from synthetic.benchmark import DEFAULT_METHODS

    found = []
    for method in DEFAULT_METHODS:
        path = out / method / f"{mode}.json"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            continue
        payload["_method"] = method
        found.append(payload)
    return found


def _problem_line(payloads: Sequence[Dict]) -> str:
    if not payloads:
        return ""
    p = payloads[0]["problem"]
    return (f"`{p['m']}x{p['n']}`, spectrum `{p['spectrum']}`, basis "
            f"`{p['basis']}`, L = {p['L']:.4g}, sigma = {p['sigma']:.4g}, "
            f"condition number = {p['condition_number']:.4g}, "
            f"lmo_dtype `{p['lmo_dtype']}`, seeds {p['problem_seeds']}\n")


def _fmt(value, spec: str = ".4g") -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and value != value:
        return "--"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def summarize(out: Path) -> str:
    """Rebuild SUMMARY.md from whatever result JSON is on disk."""
    lines: List[str] = ["# Synthetic benchmark results", ""]

    manifest_path = out / "MANIFEST.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        env = man.get("environment", {})
        lines += [
            f"Run {man.get('started', '?')} — "
            f"{env.get('gpu', 'CPU')}, torch {env.get('torch', '?')}, "
            f"commit `{(env.get('git_commit') or '?')[:10]}`"
            f"{' (dirty tree)' if env.get('git_dirty') else ''}", ""]
        rows = ["| stage | exit | minutes |", "| :--- | :--- | ---: |"]
        for st in man.get("stages", []):
            rows.append(f"| {st['stage']} | {st['exit_code']} | "
                        f"{st['seconds'] / 60:.1f} |")
        lines += rows + [""]

    # -- grid / final ---------------------------------------------------
    for mode, title in (("grid", "Tuned comparison (`tab:synthetic_results`, "
                                 "`tab:grid_search`)"),
                        ("grid-paper", "The same, on Table 3's published grids "
                                       "(`--grid-preset paper`)"),
                        ("final", "Re-run at the tuned optima "
                                  "(`fig:synthetic_results`)")):
        payloads = _load(out, mode)
        if not payloads:
            continue
        lines += [f"## {title}", "", _problem_line(payloads),
                  "| method | iterations to target | best F | min ‖∇F‖ | lr | momentum | schedule | on grid boundary |",
                  "| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |"]
        for p in payloads:
            r = p["result"]
            it = (f"{r['iters_to_converge']:.0f}" if r["reached_target"]
                  else f">{p['problem']['max_iters']}")
            edge = ", ".join(r.get("on_grid_boundary") or []) or "—"
            lines.append(
                f"| {p['_method']} | {it} | {_fmt(r['best_f'], '.3e')} | "
                f"{_fmt(r['best_gnorm'], '.3e')} | {_fmt(r['kwargs']['lr'])} | "
                f"{_fmt(r['kwargs'].get('momentum'))} | {r.get('schedule', '?')} | "
                f"{edge} |")
        lines.append("")

    # -- alignment ------------------------------------------------------
    payloads = _load(out, "alignment")
    if payloads:
        lines += ["## Gradient/step alignment", "",
                  "`rho_t = <grad F(X_t), d_t> / (||grad F(X_t)||_F ||d_t||_F)`. "
                  "The descent lemma needs it positive; the divergence theorems "
                  "make it negative. The reference column is the closed form for "
                  "`d = compressor(grad F)` at `X_0` with no momentum, so it is "
                  "only comparable to a row whose tuned momentum is 0.", "",
                  _problem_line(payloads),
                  "| method | min rho | 1st pct | median | mean | % negative | closed form | tuned |",
                  "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"]
        for p in payloads:
            r = p["result"]
            rho = r.get("rho")
            cfg = (f"eta={_fmt(r['kwargs']['lr'])}, "
                   f"mu={_fmt(r['kwargs'].get('momentum', 0))}")
            ref = _fmt(r.get("rho_reference_at_X0"), ".4f")
            if not rho:
                lines.append(f"| {p['_method']} | — | — | — | — | — | {ref} | {cfg} |")
                continue
            lines.append(
                f"| {p['_method']} | {rho['min']:.4f} | {rho['p01']:.4f} | "
                f"{rho['median']:.4f} | {rho['mean']:.4f} | "
                f"{100 * rho['frac_negative']:.2f}% | {ref} | {cfg} |")
        lines.append("")

    # -- floor ----------------------------------------------------------
    payloads = _load(out, "floor")
    if payloads:
        lines += ["## Accuracy floor of a constant step", "",
                  "Slope of `log(plateau)` against `log(eta)`. The descent lemma "
                  "predicts the gradient floor is linear in `eta` (slope 1) with "
                  "coefficient `L||s||_F^2 / (2 rho)`. SignMuon and SignSGD share "
                  "`||s||_F^2 = mn` exactly, so any gap between their floors is "
                  "attributable to `rho` alone.", "", _problem_line(payloads),
                  "| method | settled points | d log‖∇F‖/d log η | R² | d log F/d log η | R² | L‖s‖²/2 |",
                  "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
        for p in payloads:
            r = p["result"]
            s = r.get("step_norm")
            pred = (p["problem"]["L"] * s * s / 2
                    if isinstance(s, (int, float)) and s == s else None)
            n = f"{r.get('n_settled', 0)}/{len(r.get('rows', []))}"
            if r.get("n_settled", 0) < 2:
                lines.append(f"| {p['_method']} | {n} | no floor | | | | "
                             f"{_fmt(pred)} |")
                continue
            lines.append(
                f"| {p['_method']} | {n} | {r['slope_gnorm']:.3f} | "
                f"{r['r2_gnorm']:.3f} | {r['slope_f']:.3f} | {r['r2_f']:.3f} | "
                f"{_fmt(pred)} |")
        lines += ["", "<details><summary>Per-η plateaus</summary>", ""]
        for p in payloads:
            lines += [f"**{p['_method']}**", "",
                      "| η | iterations | F∞ | ‖∇F‖∞ | settled |",
                      "| ---: | ---: | ---: | ---: | :--- |"]
            for row in p["result"].get("rows", []):
                lines.append(
                    f"| {_fmt(row['lr'])} | {row.get('iters', '?')} | "
                    f"{_fmt(row['f_inf'], '.4e')} | {_fmt(row['g_inf'], '.4e')} | "
                    f"{'yes' if row.get('settled') else 'no'} |")
            lines.append("")
        lines += ["</details>", ""]

    # -- horizon --------------------------------------------------------
    payloads = _load(out, "horizon")
    if payloads:
        lines += ["## Budget scaling", "",
                  "Tuned separately at each budget `T`, then fitted. "
                  "`p = q = 1/2` is the nonconvex L-smooth bound the paper's "
                  "theorems prove; `p = q = 1` is what a strongly convex problem "
                  "gives, because the rate term contracts geometrically and the "
                  "error is floor-limited at `eta ~ 1/T`. SGD has no floor, so no "
                  "power law fits it.", "", _problem_line(payloads),
                  "| method | p (‖∇F‖) | R² | p (F) | R² | q (η*) | R² |",
                  "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for p in payloads:
            r = p["result"]
            lines.append(
                f"| {p['_method']} | {_fmt(r['exponent_gnorm'], '.3f')} | "
                f"{_fmt(r['r2_gnorm'], '.3f')} | {_fmt(r['exponent_f'], '.3f')} | "
                f"{_fmt(r['r2_f'], '.3f')} | {_fmt(r['exponent_lr'], '.3f')} | "
                f"{_fmt(r['r2_lr'], '.3f')} |")
        lines += ["", "<details><summary>Per-budget optima</summary>", ""]
        for p in payloads:
            lines += [f"**{p['_method']}**", "",
                      "| T | η* | momentum | schedule | best F | min ‖∇F‖ |",
                      "| ---: | ---: | ---: | :--- | ---: | ---: |"]
            for row in p["result"].get("rows", []):
                lines.append(
                    f"| {row['T']} | {_fmt(row['lr'])} | "
                    f"{_fmt(row.get('momentum'))} | {row.get('schedule', '?')} | "
                    f"{_fmt(row['best_f'], '.4e')} | "
                    f"{_fmt(row['best_gnorm'], '.4e')} |")
            lines.append("")
        lines += ["</details>", ""]

    # -- stability ------------------------------------------------------
    payloads = _load(out, "stability")
    if payloads:
        L = payloads[0]["problem"]["L"]
        lines += ["## Step-size stability edge", "",
                  f"L = {L:.4g}, so `2/L` = {2 / L:.4g}. **SGD is the control** "
                  "— its `eta_max` must land on `2/L`. If the operative trust "
                  "region were the Frobenius ball, `eta_max*||s||_F` would be "
                  "family-independent and also near `2/L`; the spread measures "
                  "how far off the Frobenius bound is for each step geometry.",
                  "", _problem_line(payloads),
                  "A `>` means the search hit its ceiling without finding an "
                  "edge, so that row is a lower bound, not a measurement. Adam "
                  "lands there by construction: its step is bounded by roughly "
                  "`lr` whatever the gradient does, so it oscillates instead of "
                  "diverging.", "",
                  "| method | η_max | ‖s‖_F | η_max·‖s‖_F | ÷ (2/L) |",
                  "| :--- | ---: | ---: | ---: | ---: |"]
        for p in payloads:
            r = p["result"]
            s = r.get("step_norm")
            eta = ("&gt; " if r.get("censored") else "") + _fmt(r["eta_max"])
            if not (isinstance(s, (int, float)) and s == s):
                lines.append(f"| {p['_method']} | {eta} | — | — | "
                             f"{r['eta_max'] / (2 / L):.3f} |")
                continue
            lines.append(
                f"| {p['_method']} | {eta} | {_fmt(s)} | "
                f"{_fmt(r['step_length'])} | {r['step_length'] / (2 / L):.3f} |")
        lines.append("")

    # -- kappa ----------------------------------------------------------
    payloads = _load(out, "kappa")
    if payloads:
        kappas = payloads[0]["result"].get("kappas", [])
        header = " | ".join(f"κ={k:g}" for k in kappas)
        lines += ["## Condition-number sweep", "",
                  "Best `‖∇F‖` reached within the budget, tuned at each κ. "
                  "Spectra are log-spaced with `L = 1`, so κ is exact.", "",
                  _problem_line(payloads),
                  f"| method | {header} | d log‖∇F‖/d log κ | R² |",
                  "| :--- |" + " ---: |" * (len(kappas) + 2)]
        for p in payloads:
            r = p["result"]
            cells = " | ".join(_fmt(row["best_gnorm"], ".3e")
                               for row in r.get("rows", []))
            lines.append(f"| {p['_method']} | {cells} | "
                         f"{_fmt(r.get('exponent_kappa'), '.3f')} | "
                         f"{_fmt(r.get('r2_kappa'), '.3f')} |")
        lines.append("")

    if len(lines) <= 2:
        lines.append("_No result files found._")
    return "\n".join(lines) + "\n"


def bundle(out: Path) -> Optional[Path]:
    """Zip the whole result tree into one file to copy off a remote GPU box.

    ``code/results/`` is gitignored, so on a remote machine there is otherwise
    no single artifact to bring back. Everything here is text, so it compresses
    to a fraction of its size even with the saved loss curves.
    """
    import zipfile

    archive = out.parent / f"{out.name}_results.zip"
    files = sorted(p for p in out.rglob("*") if p.is_file() and p != archive)
    if not files:
        return None
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(out.parent))
    return archive


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None,
                   help="output directory (default: results/synthetic/)")
    p.add_argument("--stages", nargs="+", default=None, metavar="NAME",
                   choices=[s.name for s in STAGES],
                   help=f"default: everything except {sorted(OPTIONAL)}")
    p.add_argument("--quick", action="store_true",
                   help="tiny sizes and coarse grids, ~5 min for the whole "
                        "pipeline; run this first")
    p.add_argument("--m", type=int, default=None, metavar="M",
                   help="Override the problem size for EVERY stage. Cost grows "
                        "steeply -- the LMO is an SVD/Newton-Schulz per step -- so "
                        "start small (--m 20) to check the pipeline and the "
                        "figures, then scale up. Unlike --quick this keeps the "
                        "real grids and iteration counts, so the numbers are "
                        "meaningful at whatever size you pick, just at a size the "
                        "paper does not report. Default: 100 for the dynamics "
                        "stages, 500 for grid/final.")
    p.add_argument("--n", type=int, default=None, metavar="N",
                   help="Columns; defaults to --m when only --m is given.")
    p.add_argument("--force", action="store_true",
                   help="re-run stages whose output already exists")
    p.add_argument("--summarize-only", action="store_true",
                   help="rebuild SUMMARY.md from the JSON on disk and exit")
    p.add_argument("--list", action="store_true", help="describe the stages and exit")
    return p.parse_args()


def already_done(stage: Stage, out: Path) -> bool:
    from synthetic.benchmark import DEFAULT_METHODS
    return any((out / m / f"{stage.out_key}.json").exists() for m in DEFAULT_METHODS)


def main() -> int:
    args = get_args()
    size = None
    if args.m is not None or args.n is not None:
        m = args.m if args.m is not None else args.n
        size = (m, args.n if args.n is not None else m)
    if args.out:
        out = Path(args.out)
    else:
        # A quick pass writes to its own tree. Sharing one would leave 64x64
        # smoke-test JSON where the real 500x500 results belong, and the
        # already-done check would then skip the real run entirely.
        out = results_root() / ("synthetic_quick" if args.quick else "synthetic")
        if size is not None:
            # A non-default size is a different experiment, not a cheaper run of
            # the same one, so it gets its own tree. Otherwise a 20x20 pass would
            # sit where the 500x500 results belong and the already-done check
            # would then skip the real run.
            out = out.with_name(f"{out.name}_{size[0]}x{size[1]}")

    if args.list:
        for s in STAGES:
            tag = "  (optional)" if s.name in OPTIONAL else ""
            print(f"\n{s.name}  [{s.estimate}]{tag}\n  {s.why}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        path = out / "SUMMARY.md"
        path.write_text(summarize(out), encoding="utf-8")
        print(f"Wrote {path.resolve()}")
        archive = bundle(out)
        if archive:
            print(f"Wrote {archive.resolve()} "
                  f"({archive.stat().st_size / 1024:.0f} KB)")
        return 0

    names = args.stages or [s.name for s in STAGES if s.name not in OPTIONAL]
    selected = [BY_NAME[n] for n in names]
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = environment(args.device)
    if not env.get("cuda_available"):
        print(f"WARNING: CUDA is not available; this will run on CPU and the "
              f"estimates below are far too optimistic.\n         {env}\n")
    else:
        print(f"{env.get('gpu')} ({env.get('gpu_memory_gb')} GB), "
              f"torch {env.get('torch')}, CUDA {env.get('cuda')}")
    if env.get("git_dirty"):
        print("NOTE: the working tree has uncommitted changes; MANIFEST.json "
              "records the commit but not those edits.")
    print(f"\nStages: {', '.join(names)}"
          f"{'   [QUICK]' if args.quick else ''}"
          f"{f'   [{size[0]}x{size[1]}]' if size else ''}"
          f"\nOutput: {out.resolve()}")

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    records: List[Dict] = []
    t0 = time.time()
    for stage in selected:
        if not args.force and not args.quick and already_done(stage, out):
            print(f"\n[{stage.name}] already has output; skipping (--force to redo)")
            records.append({"stage": stage.name, "mode": stage.mode,
                            "exit_code": 0, "seconds": 0.0, "skipped": True})
            continue
        records.append(run_stage(stage, args.device, out, log_dir, args.quick, size))

        manifest = {"started": started, "quick": args.quick,
                    "size": list(size) if size else None,
                    "environment": env, "stages": records}
        with open(out / "MANIFEST.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        (out / "SUMMARY.md").write_text(summarize(out), encoding="utf-8")

    failed = [r["stage"] for r in records if r["exit_code"] != 0]
    print(f"\n{'=' * 78}\nTotal {(time.time() - t0) / 60:.1f} min. "
          f"{len(records) - len(failed)}/{len(records)} stages ok.")
    if failed:
        print(f"FAILED: {', '.join(failed)}  (see {log_dir})")

    archive = bundle(out)
    print(f"\nRead    {(out / 'SUMMARY.md').resolve()}")
    print(f"Logs    {log_dir.resolve()}")
    if archive:
        print(f"Bundle  {archive.resolve()} "
              f"({archive.stat().st_size / 1024:.0f} KB)")
        print("\nOn a remote box, copy the bundle back and unzip it into "
              "code/results/;\notherwise just open SUMMARY.md.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
