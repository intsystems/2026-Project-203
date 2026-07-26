"""Turn modded-nanogpt run logs into tidy data.

A speedrun log is a self-contained artefact: it starts with a verbatim copy of
the training script and of ``signmuon_optimizers.py`` (so the run reproduces
itself), then a JSON ``RUNMETA`` header, then one line per step. Excellent for
provenance, useless for plotting. This module extracts the numbers.

Outputs (into ``-o OUTDIR``, default ``results/``):

``runs.csv``
    One row per run: optimizer, lr, momentum, weight decay, lr-scaling rule,
    world size, whether it diverged, final/best validation loss, total training
    time, ms/step, peak memory, and the time-to-target ("speedrun") numbers.
``steps.csv``
    Long/tidy: ``run_id, optimizer, lr, step, wallclock_ms, train_loss, val_loss``.
    One row per logged step; ``train_loss`` and ``val_loss`` are sparse (a step
    has one, the other, or both). This is the frame the plots consume.
``runs.json``
    The same as ``runs.csv`` plus the full RUNMETA/RUNEND dicts.

Wall-clock note
---------------
Two clocks appear in a log and they are NOT the same thing:

* ``train_time:`` on a *val_loss* line is the authoritative cumulative training
  time, with validation excluded -- the number the speedrun reports.
* ``train_time:`` on a plain step line is ``approx_training_time_ms``, measured
  before the step's own CUDA work has necessarily completed.

Validation-point times are therefore used as anchors, and per-step times are
kept as-is for the fine-grained curve. Both are "training time", i.e. evaluation
and compilation are excluded, which is what makes loss-vs-time comparable across
optimizers.

Usage
-----
    python parse_logs.py logs -o results
    python parse_logs.py logs/Muon_lr0.06_ab12cd34.txt logs/SignMuon_*.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

__all__ = ["parse_log", "parse_many", "write_csv", "RunRecord"]

# step:1234/2330 val_loss:3.2812 train_time:141234ms step_avg:60.61ms
_VAL_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<val>[-\deE.]+|nan|inf|-inf)\s+"
    r"train_time:(?P<ms>[\d.]+)ms")
# step:1234/2330 train_time:141234ms step_avg:60.61ms
_STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<total>\d+)\s+train_time:(?P<ms>[\d.]+)ms")
# step:1233 train_loss:3.281234
_TRAIN_RE = re.compile(
    r"^step:(?P<step>\d+)\s+train_loss:(?P<loss>[-\deE.]+|nan|inf|-inf)\s*$")
_META_RE = re.compile(r"^RUNMETA (?P<json>\{.*\})\s*$")
_END_RE = re.compile(r"^RUNEND (?P<json>\{.*\})\s*$")
_DIVERGED_RE = re.compile(r"^DIVERGED step:(?P<step>\d+)/")
_PEAK_RE = re.compile(r"^peak memory allocated: (?P<mib>\d+) MiB")
# fallback for logs written before RUNMETA existed
_OPTLINE_RE = re.compile(r"^hidden-matrix optimizer: (?P<opt>\S+)\s+config=(?P<cfg>\{.*\})")
_GPU_RE = re.compile(r"\|\s+\d+\s+(?P<name>NVIDIA [A-Za-z0-9 \-]+?)\s{2,}")


def _f(x: str) -> float:
    return float(x)


class RunRecord(dict):
    """One run: scalar summary fields plus the per-step series."""

    @property
    def steps(self) -> list[dict[str, Any]]:
        return self["_steps"]


def parse_log(path: Path) -> RunRecord | None:
    """Parse one log file. Returns ``None`` if it contains no step lines
    (e.g. a run that died during compilation)."""
    meta: dict[str, Any] = {}
    end: dict[str, Any] = {}
    train_loss: dict[int, float] = {}
    val: dict[int, tuple[float, float]] = {}     # step -> (val_loss, cumulative ms)
    step_ms: dict[int, float] = {}               # step -> approx cumulative ms
    diverged_at: int | None = None
    peak_mib: int | None = None
    gpu: str | None = None
    total_steps: int | None = None

    seps = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            # The log opens with a verbatim dump of the training script and of
            # signmuon_optimizers.py, whose own text contains every one of these
            # patterns inside f-strings. Two things keep the source from being
            # read as data: every regex is anchored at ^ (in the source those
            # patterns are always inside an indented `print0(f"...")`), and the
            # per-step regexes stay switched off until the preamble is over --
            # marked by RUNMETA, or, in logs written before RUNMETA existed, by
            # the third "="*100 separator that closes the environment dump.
            if line.startswith("=" * 20):
                seps += 1
                continue
            if not meta:
                m = _META_RE.match(line)
                if m:
                    meta = json.loads(m.group("json"))
                    continue
                m = _OPTLINE_RE.match(line)      # pre-RUNMETA logs
                if m:
                    cfg = m.group("cfg").replace("'", '"')
                    try:
                        meta = dict(json.loads(cfg), optimizer=m.group("opt"))
                    except json.JSONDecodeError:
                        meta = dict(optimizer=m.group("opt"))
                    continue
            if gpu is None:
                g = _GPU_RE.search(line)
                if g:
                    gpu = g.group("name").strip()
                    continue
            if not meta and seps < 3:
                continue                          # still inside the source dump

            m = _TRAIN_RE.match(line)
            if m:
                train_loss[int(m.group("step"))] = _f(m.group("loss"))
                continue
            m = _VAL_RE.match(line)
            if m:
                total_steps = int(m.group("total"))
                val[int(m.group("step"))] = (_f(m.group("val")), _f(m.group("ms")))
                continue
            m = _STEP_RE.match(line)
            if m:
                total_steps = int(m.group("total"))
                step_ms[int(m.group("step"))] = _f(m.group("ms"))
                continue
            m = _DIVERGED_RE.match(line)
            if m:
                diverged_at = int(m.group("step"))
                continue
            m = _PEAK_RE.match(line)
            if m:
                peak_mib = int(m.group("mib"))
                continue
            m = _END_RE.match(line)
            if m:
                end = json.loads(m.group("json"))

    if not (train_loss or val):
        return None

    steps = sorted(set(train_loss) | set(val) | set(step_ms))
    series = []
    for s in steps:
        v = val.get(s)
        # Prefer the validation-line clock (validation excluded, authoritative);
        # fall back to the approximate per-step clock.
        ms = v[1] if v is not None else step_ms.get(s)
        series.append(dict(step=s, wallclock_ms=ms,
                           train_loss=train_loss.get(s),
                           val_loss=None if v is None else v[0]))

    finite_val = {s: v for s, (v, _) in val.items() if v == v and abs(v) != float("inf")}
    rec = RunRecord(
        run_id=meta.get("run_id", path.stem),
        log=str(path),
        optimizer=meta.get("optimizer"),
        family=meta.get("family"),
        lr=meta.get("lr"),
        momentum=meta.get("momentum"),
        weight_decay=meta.get("weight_decay"),
        lr_scaling=meta.get("lr_scaling"),
        world_size=meta.get("world_size"),
        train_steps=meta.get("train_steps", total_steps),
        tokens_per_step=meta.get("tokens_per_step"),
        gpu=gpu,
        diverged=bool(end.get("diverged", diverged_at is not None)),
        diverged_at=diverged_at,
        completed=bool(end),
        last_step=max(steps) if steps else None,
        final_val_loss=finite_val.get(max(finite_val)) if finite_val else None,
        best_val_loss=min(finite_val.values()) if finite_val else None,
        train_time_ms=end.get("train_time_ms",
                              max((m for m in step_ms.values()), default=None)),
        peak_memory_mib=end.get("peak_memory_mib", peak_mib),
    )
    n = rec["last_step"] or 0
    rec["ms_per_step"] = round(rec["train_time_ms"] / n, 3) if rec["train_time_ms"] and n else None
    rec["_steps"] = series
    rec["_meta"] = meta
    rec["_end"] = end
    return rec


def time_to_loss(rec: RunRecord, target: float) -> float | None:
    """Training milliseconds until validation loss first reaches ``target``.

    Linearly interpolates between the two bracketing validation points, so the
    number does not depend on where the (coarse) validation grid happens to fall.
    """
    pts = [(s["wallclock_ms"], s["val_loss"]) for s in rec.steps
           if s["val_loss"] is not None and s["wallclock_ms"] is not None]
    prev = None
    for ms, v in pts:
        if v <= target:
            if prev is None:
                return ms
            pms, pv = prev
            if pv == v:
                return ms
            frac = (pv - target) / (pv - v)
            return pms + frac * (ms - pms)
        prev = (ms, v)
    return None


def steps_to_loss(rec: RunRecord, target: float) -> float | None:
    """Optimizer steps until validation loss first reaches ``target``."""
    pts = [(s["step"], s["val_loss"]) for s in rec.steps if s["val_loss"] is not None]
    prev = None
    for st, v in pts:
        if v <= target:
            if prev is None:
                return float(st)
            pst, pv = prev
            if pv == v:
                return float(st)
            return pst + (pv - target) / (pv - v) * (st - pst)
        prev = (st, v)
    return None


def parse_many(paths: Iterable[Path]) -> list[RunRecord]:
    out = []
    for p in sorted(paths):
        try:
            rec = parse_log(p)
        except Exception as exc:                      # a truncated log must not kill the batch
            print(f"  !! {p.name}: {type(exc).__name__}: {exc}")
            continue
        if rec is None:
            print(f"  -- {p.name}: no step lines, skipped")
            continue
        out.append(rec)
    return out


_RUN_FIELDS = ["run_id", "optimizer", "family", "lr", "momentum", "weight_decay",
               "lr_scaling", "world_size", "gpu", "train_steps", "last_step",
               "diverged", "completed", "final_val_loss", "best_val_loss",
               "train_time_ms", "ms_per_step", "peak_memory_mib", "log"]


def write_csv(records: list[RunRecord], outdir: Path, targets: list[float]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fields = list(_RUN_FIELDS)
    for t in targets:
        fields += [f"steps_to_{t:g}", f"ms_to_{t:g}"]
    with (outdir / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {k: r.get(k) for k in _RUN_FIELDS}
            for t in targets:
                row[f"steps_to_{t:g}"] = steps_to_loss(r, t)
                row[f"ms_to_{t:g}"] = time_to_loss(r, t)
            w.writerow(row)

    with (outdir / "steps.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "optimizer", "lr", "lr_scaling", "step",
                    "wallclock_ms", "train_loss", "val_loss"])
        for r in records:
            for s in r.steps:
                w.writerow([r["run_id"], r["optimizer"], r["lr"], r["lr_scaling"],
                            s["step"], s["wallclock_ms"], s["train_loss"], s["val_loss"]])

    with (outdir / "runs.json").open("w", encoding="utf-8") as fh:
        json.dump([{k: v for k, v in r.items() if k != "_steps"} for r in records],
                  fh, indent=2, default=str)


def _collect(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += sorted(p.rglob("*.txt"))
        elif p.exists():
            paths.append(p)
        else:                                          # let the shell off the hook
            paths += sorted(Path().glob(item))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="log files and/or directories of logs")
    ap.add_argument("-o", "--outdir", default="results", type=Path)
    ap.add_argument("--target", type=float, nargs="*", default=[3.28],
                    help="val-loss targets for the time-to-loss columns "
                         "(default 3.28, the speedrun's own target)")
    args = ap.parse_args()

    paths = _collect(args.inputs)
    if not paths:
        raise SystemExit(f"no logs found in {args.inputs}")
    print(f"parsing {len(paths)} log file(s)...")
    records = parse_many(paths)
    if not records:
        raise SystemExit("no parsable runs")
    write_csv(records, args.outdir, args.target)

    w = max(len(str(r["optimizer"])) for r in records)
    print(f"\n{'optimizer':<{w}} {'lr':>7} {'steps':>6} {'final val':>10} "
          f"{'best val':>9} {'time':>9} {'ms/step':>8}  status")
    for r in sorted(records, key=lambda r: (r["final_val_loss"] is None,
                                            r["final_val_loss"] or 0)):
        t = r["train_time_ms"]
        print(f"{str(r['optimizer']):<{w}} {r['lr'] or float('nan'):>7.4g} "
              f"{r['last_step'] or 0:>6} "
              f"{(r['final_val_loss'] if r['final_val_loss'] is not None else float('nan')):>10.4f} "
              f"{(r['best_val_loss'] if r['best_val_loss'] is not None else float('nan')):>9.4f} "
              f"{(t / 1000 if t else float('nan')):>8.1f}s "
              f"{(r['ms_per_step'] or float('nan')):>8.2f}  "
              f"{'DIVERGED' if r['diverged'] else ('ok' if r['completed'] else 'incomplete')}")
    print(f"\nwrote {args.outdir}/runs.csv, {args.outdir}/steps.csv, {args.outdir}/runs.json")


if __name__ == "__main__":
    main()
