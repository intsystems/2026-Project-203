#!/usr/bin/env python3
"""Build and audit the formalization using Lake from PATH on any platform."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
PIN_FILES = ("lean-toolchain", "lakefile.toml", "lake-manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparator",
        action="store_true",
        help="also compare frozen statements and replay proofs using LongTails' Comparator",
    )
    parser.add_argument(
        "--fetch-cache",
        action="store_true",
        help="download Mathlib's compiled cache before building",
    )
    args = parser.parse_args()

    lake = shutil.which("lake")
    if lake is None:
        print("error: lake is not on PATH. Install Elan first.", file=sys.stderr)
        return 2

    try:
        pins = {name: (PROJECT / name).read_bytes() for name in PIN_FILES}
    except OSError as error:
        print("error: cannot read the committed dependency pins: {}".format(error), file=sys.stderr)
        return 2

    commands = []
    if args.fetch_cache:
        commands.append(("exe", "cache", "get"))
    commands.extend((("build",), ("env", "lean", "Audit.lean")))

    status = 0
    try:
        for command in commands:
            print("+ lake " + " ".join(command), flush=True)
            result = subprocess.run([lake, *command], cwd=PROJECT)
            if result.returncode:
                print("error: verification stopped after a failed Lake command.", file=sys.stderr)
                status = 1
                break
        if status == 0 and args.comparator:
            result = subprocess.run([sys.executable, str(PROJECT / "scripts" / "comparator.py")],
                                    cwd=PROJECT)
            status = int(result.returncode != 0)
    except OSError as error:
        print("error: cannot run Lake: {}".format(error), file=sys.stderr)
        status = 1

    for name, original in pins.items():
        try:
            unchanged = (PROJECT / name).read_bytes() == original
        except OSError:
            unchanged = False
        if not unchanged:
            print("error: verification changed dependency pin file: " + name, file=sys.stderr)
            status = 1

    if status == 0:
        checks = "library build, axiom audit, and dependency pins"
        if args.comparator:
            checks += ", plus Comparator"
        print("Verification passed: " + checks + ".")
    return status


if __name__ == "__main__":
    sys.exit(main())
