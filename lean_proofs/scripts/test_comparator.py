#!/usr/bin/env python3
"""Exercise the real Comparator adapter on a valid proof and two invalid submissions."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from comparator import ALLOWED, PRIMITIVES, PROJECT, run


def main():
    lake = shutil.which("lake")
    if lake is None:
        raise RuntimeError("lake is not on PATH")
    root = PROJECT / ".comparator"
    sources = root / "ComparatorChallenge"
    sources.mkdir(parents=True, exist_ok=True)
    statements = {
        "TestReference": "theorem ComparatorTest.result : (2 + 2 : Nat) = 4 := by sorry\n",
        "TestGood": "theorem ComparatorTest.result : (2 + 2 : Nat) = 4 := rfl\n",
        "TestBadType": "theorem ComparatorTest.result : (2 + 2 : Nat) = 5 := by sorry\n",
        "TestBadAxiom": "axiom ComparatorTest.extra : (2 + 2 : Nat) = 4\n"
                        "theorem ComparatorTest.result : (2 + 2 : Nat) = 4 := ComparatorTest.extra\n",
    }
    for name, source in statements.items():
        (sources / (name + ".lean")).write_text("import Init\n" + source, encoding="utf-8")
    modules = ["ComparatorChallenge." + name for name in statements]
    run(lake, ["build", "comparator", "lean4export", *modules])
    executable = "lean4export.exe" if os.name == "nt" else "lean4export"
    exporter = PROJECT / ".lake/packages/lean4export/.lake/build/bin" / executable
    constants = ["ComparatorTest.result", *ALLOWED, *PRIMITIVES]
    with tempfile.TemporaryDirectory(prefix="test-", dir=root) as directory:
        work = Path(directory).resolve()
        if PROJECT not in work.parents:
            raise ValueError("temporary test directory escaped the project")
        for module in modules:
            with (work / (module + ".ndjson")).open("wb") as output:
                run(lake, ["env", exporter, module, "--", *constants], output=output)
        for name, accepted in [("TestGood", True), ("TestBadType", False), ("TestBadAxiom", False)]:
            solution = "ComparatorChallenge." + name
            config = {
                "challenge_module": modules[0], "solution_module": solution,
                "theorem_names": ["ComparatorTest.result"], "permitted_axioms": ALLOWED,
            }
            config_path = work / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "EF21MUON_COMPARATOR_CONFIG": str(config_path),
                "EF21MUON_COMPARATOR_CHALLENGE": str(work / (modules[0] + ".ndjson")),
                "EF21MUON_COMPARATOR_SOLUTION": str(work / (solution + ".ndjson")),
            })
            result = subprocess.run([lake, "env", "lean", "scripts/CompareExports.lean"],
                                    cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
            if (result.returncode == 0) != accepted:
                raise RuntimeError(name + " produced the wrong verdict:\n" + result.stdout)
            if not accepted:
                diagnostic = "statement do not match" if name == "TestBadType" else "axiom"
                if diagnostic not in result.stdout.lower():
                    raise RuntimeError(name + " failed for an unexpected reason:\n" + result.stdout)
            print(name + ": " + ("accepted" if accepted else "rejected as expected"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit("Comparator self-test failed: " + str(error))
