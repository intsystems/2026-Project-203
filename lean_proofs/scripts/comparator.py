#!/usr/bin/env python3
"""Compare trusted, prebuilt proofs with frozen statements using pinned Comparator."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
CATALOG = PROJECT / "comparator" / "targets.json"
REFERENCE = PROJECT / "comparator" / "Reference.lean.in"
ALLOWED = ["propext", "Classical.choice", "Quot.sound"]
PRIMITIVES = [
    "Nat.add", "Nat.sub", "Nat.mul", "Nat.pow", "Nat.gcd", "Nat.div", "Nat.mod",
    "Nat.beq", "Nat.ble", "Nat.land", "Nat.lor", "Nat.xor", "Nat.shiftLeft",
    "Nat.shiftRight", "String.ofList", "Char.ofNat", "List", "eagerReduce", "Nat",
    "String", "String.mk", "Char", "optParam", "autoParam", "semiOutParam", "outParam",
    "Quot", "Quot.mk", "Quot.lift", "Quot.ind",
]


def run(lake, arguments, *, env=None, output=None):
    print("+ lake " + " ".join(map(str, arguments)), flush=True)
    result = subprocess.run([lake, *map(str, arguments)], cwd=PROJECT, env=env,
                            stdout=output, stderr=None if output is None else subprocess.PIPE)
    if result.returncode:
        if result.stderr:
            sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError("Lake command failed with exit code {}".format(result.returncode))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        help="write a JSON verification report (default: .comparator/report.json)")
    args = parser.parse_args()
    work_root = PROJECT / ".comparator"
    report_path = args.report or work_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"passed": False, "status": "incomplete"}) + "\n",
                           encoding="utf-8")
    lake = shutil.which("lake")
    if lake is None:
        parser.error("lake is not on PATH")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    targets = [entry["name"] for entry in catalog["targets"]]
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("the frozen target catalog must contain distinct declarations")
    reference = REFERENCE.read_bytes()
    challenge = work_root / "ComparatorChallenge" / "Reference.lean"
    challenge.parent.mkdir(parents=True, exist_ok=True)
    challenge.write_bytes(reference)
    run(lake, ["build", "EF21Muon.ComparatorTargets", "comparator", "lean4export"])
    run(lake, ["build", "ComparatorChallenge.Reference"])
    executable = "lean4export.exe" if os.name == "nt" else "lean4export"
    exporter = PROJECT / ".lake" / "packages" / "lean4export" / ".lake" / "build" / "bin" / executable
    if not exporter.is_file():
        raise ValueError("built lean4export executable not found: " + str(exporter))
    config = {
        "challenge_module": "ComparatorChallenge.Reference",
        "solution_module": "EF21Muon.ComparatorTargets",
        "theorem_names": targets,
        "permitted_axioms": ALLOWED,
    }
    constants = list(dict.fromkeys(targets + ALLOWED + PRIMITIVES))
    with tempfile.TemporaryDirectory(prefix="run-", dir=work_root) as directory:
        work = Path(directory).resolve()
        if PROJECT not in work.parents:
            raise ValueError("temporary audit directory escaped the project")
        config_path = work / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        challenge_export = work / "challenge.ndjson"
        solution_export = work / "solution.ndjson"
        for module, destination in [(config["challenge_module"], challenge_export),
                                    (config["solution_module"], solution_export)]:
            with destination.open("wb") as output:
                run(lake, ["env", exporter, module, "--", *constants], output=output)
        env = os.environ.copy()
        env["EF21MUON_COMPARATOR_CONFIG"] = str(config_path)
        env["EF21MUON_COMPARATOR_CHALLENGE"] = str(challenge_export)
        env["EF21MUON_COMPARATOR_SOLUTION"] = str(solution_export)
        run(lake, ["env", "lean", "scripts/CompareExports.lean"], env=env)
    report = {
        "passed": True,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "checked_statements": len(targets),
        "targets": catalog["targets"],
        "permitted_axioms": ALLOWED,
        "reference_sha256": hashlib.sha256(reference).hexdigest(),
        "dependency_pins_sha256": {
            name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
            for name in ["lean-toolchain", "lakefile.toml", "lake-manifest.json"]
        },
        "mode": "Comparator comparison and kernel replay of trusted, prebuilt sources",
        "hostile_source_sandbox": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Comparator audit passed for {} frozen statements.".format(len(targets)))
    print("Report: " + str(report_path))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit("Comparator audit failed: " + str(error))
