# Verification tools

## Universal checker

Install Python 3, Git, and [Elan](https://github.com/leanprover/elan#installation),
with `lake` on PATH. From the repository root, run:

```sh
python lean_proofs/scripts/check.py --fetch-cache
```

Use `python3` if that is the name of your Python 3 interpreter. This same script
works on Linux, macOS, and Windows and locates the project relative to itself.
It inherits the normal environment of the installed toolchain.

The checker:

1. Reads the committed toolchain and dependency pins.
2. With `--fetch-cache`, downloads Mathlib's compiled cache using Lake.
3. Builds the library with `lake build`.
4. Enforces the axiom allowlist with `lake env lean Audit.lean`.
5. With `--comparator`, compares the frozen statements and replays their proofs.
6. Checks that all dependency pin files remain unchanged.

It stops on a failed command and returns a nonzero exit status on failure.
Omit `--fetch-cache` for subsequent checks. It requires only Python's standard
library; dependencies and build artifacts are managed by Lake.

## LongTails Comparator

```sh
python lean_proofs/scripts/check.py --comparator
```

Or run just the Comparator audit:

```sh
python lean_proofs/scripts/comparator.py
```

This builds the pinned Comparator and lean4export, exports the frozen reference
and proved declarations, then calls Comparator's statement comparison, axiom
check, primitive checks, and Lean kernel replay. The reference is copied from
`comparator/Reference.lean.in`; it is never inferred from the current solutions.
Temporary challenge sources and exports stay in ignored `.comparator/`.
The default report is `.comparator/report.json`; `--report PATH` selects another
location. A failed run cannot retain a successful status from an earlier run.

This path supports native Windows as well as POSIX systems. It builds trusted
repository sources directly and does not provide LongTails' Linux sandbox for
untrusted proof submissions. See [RESULTS.md](../RESULTS.md#longtails-audit-and-accuracy-tuning)
for certificate coverage and oracle accounting.

`python lean_proofs/scripts/test_comparator.py` exercises the adapter with a
valid proof, a mismatching statement, and an unapproved axiom. The first must be
accepted and both invalid submissions must be rejected by Comparator. These
fixtures are isolated from the library and its frozen scientific statements.

## Bound calculator

The optional calculator also uses only Python's standard library:

```sh
python lean_proofs/scripts/compare_bounds.py --rows 32 --cols 32 --beta 0.1
```

See the [main README](../README.md) and [mathematical notes](../RESULTS.md) for
scope and interpretation.
