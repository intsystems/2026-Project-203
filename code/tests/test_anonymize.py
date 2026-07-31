"""Anonymity checks: the scan, the bundler, and what the exporters must not write.

    python3 -m tests.test_code      # runs these too, when this file is present
    pytest tests/test_anonymize.py  # or on their own

Separate from `test_code.py` for one reason: this module and the `anonymize.py` it
imports are both withheld from the anonymous bundle -- the scanner's content is the
list of names that must not ship, and these tests quote its rules. Keeping them
here means the test suite a reviewer runs has no import that cannot resolve, and no
skips to explain, while a working tree still runs them on every preflight.

`test_code.py` picks the tests up by importing this module when it exists, so
nothing here needs its own runner.
"""

from __future__ import annotations

def test_the_export_bundle_carries_no_home_path():
    """A bundle exported from a home-rooted tree must be anonymous.

    This matters more now that `results/` ships with the code: the bundle is no
    longer only a file you download, it is a file reviewers read. The exporter runs
    on the GPU box, where the tree is `/home/<user>/SignMuon/code/results/...`, and
    that string reached `runs.csv` (the `path` column) and `MANIFEST.json` (`argv`
    records whatever --root was given) even after `configs.json` and the driver
    state had been scrubbed.

    Checked with `anonymize`'s own rules rather than a local regex, so the test and
    the scanner cannot drift apart.

    The home path is written with forward slashes rather than built from the temp
    directory. Built from it, this test passed on Windows and failed on Linux: the
    fixture came out `C:\\...\\home\\someuser\\...`, which the home-path rule cannot
    match, so the `datadir` leak it was meant to catch was invisible on the machine
    the test was written on and stopped the GPU box's preflight instead.
    """
    import json
    import tempfile
    from pathlib import Path

    import anonymize
    from federated import export_article

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "home" / "someuser" / "SignMuon" / "code" / "results" / "federated"
        for seed in range(2):
            d = root / "final_signmuon_unit-gain_e2000_f" / f"seed{seed}"
            d.mkdir(parents=True)
            cfg = {"algorithm": "signmuon", "lr": 0.05, "seed": seed, "rounds": 200,
                   "split": "full", "weight_decay": 0.0, "last_k": 2,
                   "target_acc": 80.0, "dataset": "cifar10", "model": "cnn2",
                   "lr_scaling": "unit-gain", "scale_baselines": False,
                   "run_name": "final_signmuon_unit-gain_e2000_f",
                   # An absolute --data, as `--data /home/<name>/datasets` gives.
                   # POSIX-style on purpose (see the docstring), and silenced with
                   # the line pragma rather than by adding it to `anonymize.ALLOW`:
                   # ALLOW is substring-based, so allowing it globally would make
                   # the scan below skip the very leak this test exists to catch.
                   "datadir": "/home/someuser/SignMuon/code/data_federated",  # anonymize: allow
                   "device": "cuda:0"}
            hist = {"steps": [0, 100, 200], "test_acc": [10.0, 80.0, 85.0]}
            (d / "metrics.json").write_text(json.dumps({"config": cfg, "history": hist}))

        out = Path(td) / "bundle"
        assert export_article.main(["--root", str(root), "--out", str(out),
                                    "--overnight", str(Path(td) / "none"),
                                    "--no-archive"]) == 0

        findings = []
        for path in sorted(out.rglob("*")):
            if not path.is_file() or not anonymize._scannable(path):
                continue
            findings += anonymize.scan_text(
                path.relative_to(out).as_posix(),
                path.read_text(encoding="utf-8", errors="replace"))
        assert not findings, ("the export bundle is not anonymous:\n  "
                             + "\n  ".join(str(f) for f in findings[:10]))


def test_no_identifying_material():
    """`code/` must stay anonymous, checked here rather than remembered.

    Author names and absolute paths arrive one docstring at a time, so the scan
    belongs in the suite that runs before every night, not in a pre-submission
    checklist that gets read once.
    """
    import anonymize

    findings, _ = anonymize.scan_tree()
    assert not findings, (
        "identifying material in code/ -- run `python3 -m anonymize --check`:\n  "
        + "\n  ".join(str(f) for f in findings[:20]))


def test_unpacked_result_bundles_are_excluded_from_scan_and_bundle():
    """A downloaded bundle unpacked next to the code is run output, not source.

    Both workflows say to unpack the archive beside `code/`, and the bundles carry
    the absolute paths of the box that wrote them -- so an unpacked one trips the
    anonymity scan and, worse, would ship inside the anonymous bundle. `article_export`
    was already excluded; a copy parked as `article_export.stale_2026-07-29` was not,
    and neither was the federated bundle, which did not exist when the rule was
    written.
    """
    import anonymize

    for rel in ("article_export/runs.csv",
                "article_export.stale_2026-07-29/overnight/state.json",
                "centralized/article_export.stale_2026-07-29/MANIFEST.md",
                "federated_export/SUMMARY.md",
                "federated_export_results.zip",
                "results/federated/run/seed0/metrics.json",
                "results_old/synthetic_prefix_grid/SUMMARY.md"):
        assert anonymize._excluded(rel), f"{rel} should be excluded"

    # ...without swallowing the source that lives beside them.
    for rel in ("federated/export_article.py", "federated/README.md",
                "REPRODUCE.md", "results/nanogpt/log.txt"):
        assert not anonymize._excluded(rel), f"{rel} must still be scanned"


def test_anonymity_scan_actually_catches_things():
    """A scan that cannot fail is not a check.

    Feeds the scanner one line per rule and requires a hit, so that a botched regex
    (or an over-broad ALLOW entry) shows up here rather than as a clean bill of
    health on a leaky bundle.

    **No real name is written out below.** The fixtures are built from
    `anonymize.PROJECT_IDENTIFIERS` and `WORD_IDENTIFIERS` at run time, because this
    file ships in the anonymous bundle and `anonymize.py` deliberately does not:
    spelling the names here would hand a reviewer the very list the bundle exists to
    withhold, labelled as such. Building them instead also tests every identifier
    rather than the two someone happened to type.
    """
    import anonymize

    # Every name on both lists must be caught, bare and glued to a filename. `\b`
    # counts `_` as a word character, so `<name>_nanogpt.py` is the shape a `\b...\b`
    # rule misses -- it is the leak that got past the first version of the scanner.
    identifiers = tuple(anonymize.PROJECT_IDENTIFIERS) + tuple(anonymize.WORD_IDENTIFIERS)
    assert identifiers, "the identifier lists are empty; the scan checks nothing"
    for ident in identifiers:
        for text in (f"# thanks to {ident.title()} for the CNN2 baseline",
                     f"from {ident}_nanogpt import Model",
                     f"| `{ident}_nanogpt.py` | superseded |",
                     f"{ident}_2026.tex",
                     f"handle: {ident}1"):
            assert anonymize.scan_text("fake.py", text), \
                f"an identifier joined by _ . or a digit must still be caught: {text}"

    # The remaining rules take invented fixtures, which would themselves trip the
    # scan on this file; the `anonymize: allow` pragma exempts them individually --
    # visibly, and without exempting the rest of the line or the file.
    for text, rule in [
        ("contact: someone@example.org", "email"),                           # anonymize: allow
        ('DATA = "/home/jdoe/project/data"', "home-path"),                    # anonymize: allow
        # A personal fork, which the identifier list alone would not catch.
        ("see https://github.com/someuser/signmuon-fork", "project-repo"),    # anonymize: allow
        ("ORCID 0000-0002-1825-0097", "orcid"),                               # anonymize: allow
    ]:
        hits = anonymize.scan_text("fake.py", text)
        assert hits, f"rule {rule!r} failed to fire on: {text}"

    # ... and upstream citations must NOT fire, or the fix would be to delete them.
    for text in ("from https://github.com/KellerJordan/modded-nanogpt",
                 "adapted from https://github.com/NoahAmsel/PolarExpress by @varunneal",
                 "    @torch.compile",
                 "the polar factor is released under a permissive licence"):
        assert not anonymize.scan_text("fake.py", text), f"false positive on: {text}"


def test_notebook_stripping_removes_outputs_not_source():
    """The bundler must clear outputs and leave the code intact."""
    import json

    import anonymize

    leak = "/home/" + "jdoe"          # assembled, so this file stays scannable
    nb = {"cells": [{"cell_type": "code", "source": [f"print('{leak}')\n"],
                     "execution_count": 7,
                     "outputs": [{"output_type": "stream", "text": [leak + "\n"]}]},
                    {"cell_type": "markdown", "source": ["# title\n"]}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    out = json.loads(anonymize.strip_notebook(json.dumps(nb)))
    assert out["cells"][0]["outputs"] == []
    assert out["cells"][0]["execution_count"] is None
    assert out["cells"][0]["source"] == [f"print('{leak}')\n"], "source must survive"
    assert out["cells"][1] == nb["cells"][1], "markdown cells are untouched"


def test_the_scan_opens_the_archives_it_ships():
    """A shipped archive must not be a hole in the gate.

    The bundle carries the three export archives, because they are what
    `REPRODUCE.md` §§3-5 tell a reviewer to plot from. They are also zips of JSON,
    CSV and Markdown written on the machine that ran the study -- the exact
    material the scan exists to catch, and invisible to it, since a zip cannot be
    read as text. Excluding them used to be the answer; now that they ship, the
    scan has to open them.
    """
    import json
    import tarfile
    import tempfile
    import zipfile
    from pathlib import Path

    import anonymize

    leak = "/home/" + "jdoe"           # assembled, so this file stays scannable
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "results").mkdir()

        with zipfile.ZipFile(root / "results/synthetic_results.zip", "w") as z:
            z.writestr("synthetic/MANIFEST.json", json.dumps({"out": leak + "/code"}))
            z.writestr("synthetic/SUMMARY.md", "# nothing identifying here\n")
        findings, _ = anonymize.scan_tree(root)
        assert [f.rule for f in findings] == ["home-path"], findings
        assert "MANIFEST.json" in findings[0].path, \
            f"the member must be named, not just the archive: {findings[0].path}"

        # A .tar.gz too -- the centralized bundle is one.
        (root / "results/synthetic_results.zip").unlink()
        payload = root / "payload.md"
        payload.write_text(f"source: {leak}/SignMuon\n")
        with tarfile.open(root / "results/article_export.tar.gz", "w:gz") as t:
            t.add(payload, arcname="article_export/MANIFEST.md")
        payload.unlink()
        findings, _ = anonymize.scan_tree(root)
        assert [f.rule for f in findings] == ["home-path"], findings

        # An archive that does NOT ship is still not opened: it is excluded from the
        # bundle, so what it holds cannot reach a reviewer, and reading every stale
        # bundle lying around the tree would fail the gate on files nobody sends.
        (root / "results/article_export.tar.gz").unlink()
        with zipfile.ZipFile(root / "results/federated_export_stale.zip", "w") as z:
            z.writestr("x.md", leak + "/x")
        findings, _ = anonymize.scan_tree(root)
        assert not findings, f"an excluded archive must not be scanned: {findings}"


def test_the_export_archives_ship_and_the_raw_trees_do_not():
    """What a reviewer needs to redraw a figure, and nothing that merely accumulated.

    The raw run trees hold runs from superseded sessions -- 66 of them in the
    federated tree at the time of writing -- and each exporter decides which of its
    runs belong in the archive it writes. Shipping the archive ships that decision;
    shipping `results/federated/` would ship the leftovers too, under names that
    make them look like the reported runs.
    """
    import anonymize

    for rel in ("results/synthetic_results.zip",
                "results/article_export.tar.gz",
                "results/federated_export_results.zip",
                "results/nanogpt/SUMMARY.md"):
        assert not anonymize._excluded(rel), f"{rel} must ship"

    for rel in ("results/federated/final_signmuon_e2000_fs0/seed0/metrics.json",
                "results/synthetic/SUMMARY.md",
                "results/article_export/runs.csv",
                "results/federated_export/SUMMARY.md",
                "results/analysis/cifar_main.pdf",
                "results/federated_tuning_logs/signmuon.log",
                "results_old/federated_2026-07-28_ternary/x.json"):
        assert anonymize._excluded(rel), f"{rel} must not ship"


def test_the_zip_and_the_directory_are_the_same_bundle():
    """`--build` and `--build-dir` must ship identical trees.

    OpenReview wants an archive, but an archive nobody has opened is how a leak
    ships: the one check `ANONYMIZATION.md` calls non-optional is grepping the built
    artifact with tools other than the builder, which is far easier on a directory.
    That is only worth anything if the directory is what gets zipped -- two build
    paths that quietly disagree about one exclusion would make the inspected object
    and the submitted one different objects.
    """
    import json
    import tempfile
    import zipfile
    from pathlib import Path

    import anonymize

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "code"
        for rel, text in (
            ("a.py", "x = 1\n"),
            ("sub/b.md", "# b\n"),
            ("night.log", "driver output\n"),
            ("bash.exe.stackdump", "Stack trace:\n"),      # a crash dump, gitignored
            ("anonymize.py", "PROJECT_IDENTIFIERS = ()\n"),
            ("__pycache__/a.cpython-39.pyc", "junk\n"),
            ("results/federated/run/seed0/metrics.json", "{}\n"),
            ("results/nanogpt/logs/run.txt", "step 1\n"),
        ):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        nb = {"cells": [{"cell_type": "code", "source": ["print(1)\n"],
                         "execution_count": 3,
                         "outputs": [{"output_type": "stream", "text": ["1\n"]}]}],
              "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        (root / "notes.ipynb").write_text(json.dumps(nb))

        zip_path = Path(td) / "supplement.zip"
        out_dir = Path(td) / "anonymous_code"
        zip_info = anonymize.build_bundle(zip_path, root)
        dir_info = anonymize.build_dir(out_dir, root)

        in_zip = sorted(zipfile.ZipFile(zip_path).namelist())
        in_dir = sorted(p.relative_to(out_dir).as_posix()
                        for p in out_dir.rglob("*") if p.is_file())
        assert in_zip == in_dir, f"zip and directory differ: {set(in_zip) ^ set(in_dir)}"
        assert zip_info["files"] == dir_info["files"]
        assert zip_info["notebooks_stripped"] == dir_info["notebooks_stripped"] == \
            ["notes.ipynb"]

        assert in_zip == ["code/MANIFEST.md", "code/a.py", "code/notes.ipynb",
                          "code/results/nanogpt/logs/run.txt", "code/sub/b.md"], in_zip

        # And the directory really is inspectable with ordinary tools: the notebook
        # output is gone on disk, not only inside the archive.
        on_disk = json.loads((out_dir / "code/notes.ipynb").read_text(encoding="utf-8"))
        assert on_disk["cells"][0]["outputs"] == []

    # Refuses to write into a directory that already holds something, rather than
    # merging this build into the last one.
    with tempfile.TemporaryDirectory() as td:
        occupied = Path(td) / "occupied"
        occupied.mkdir()
        (occupied / "leftover.txt").write_text("from an earlier build\n")
        try:
            anonymize.build_dir(occupied, Path(td))
        except FileExistsError:
            pass
        else:
            raise AssertionError("build_dir must refuse a non-empty destination")


def test_the_bundle_records_no_git_revision():
    """No shipped file, and no member of a shipped archive, names a commit.

    The one identifying field the regex gate cannot argue about: a 40-hex SHA is
    not a name, an email or a path, so nothing in `PATTERNS` fires on it -- and it
    was the current HEAD of a repository named after the project. It resolves to
    nothing for a reviewer, who has the code and not the repository, so redacting
    it costs the supplement no reproducibility at all.

    Checked on what the builder writes, and separately on the redactor, so a
    regression in either shows up here.
    """
    import io
    import re
    import tarfile
    import zipfile

    import anonymize

    # The redactor: the fields go, a run identifier that merely looks like hex
    # stays, and so does the surrounding text.
    before = ('{"git_commit": "1feeb2a758cefda41013d7378e4ca01de132346c",\n'
              ' "commit": "cf18382136", "git_dirty": true,\n'
              ' "log": "logs/EF21-MuonSign_lr0.06_e6770317.txt"}\n'
              "Run 2026-07-29 -- commit `cf18382136`\n")
    after = anonymize.redact_revisions(before)
    assert "1feeb2a" not in after and "cf18382136" not in after, after
    assert after.count(anonymize.REDACTED_REVISION) == 3, after
    assert "e6770317" in after, "a run identifier is not a revision"
    assert '"git_dirty": true' in after, "only the revision field is rewritten"

    # And the bundle as it will ship.
    SHA = re.compile(r"\b[0-9a-f]{7,40}\b")

    def members(name, data):
        low = name.lower()
        if low.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    if not info.is_dir():
                        yield from members(f"{name}:{info.filename}", z.read(info))
        elif low.endswith((".tar.gz", ".tgz")):
            with tarfile.open(fileobj=io.BytesIO(data)) as t:
                for m in t.getmembers():
                    if m.isfile():
                        yield from members(f"{name}:{m.name}",
                                           t.extractfile(m).read())
        elif anonymize._scannable(Path(name)):
            yield name, data.decode("utf-8", "replace")

    from pathlib import Path

    leaks = []
    for rel, payload, _ in anonymize._bundle_entries(anonymize.ROOT):
        for name, text in members(rel.as_posix(), payload):
            for i, line in enumerate(text.splitlines(), 1):
                if not re.search(r"(?i)commit|revision|\bsha\b", line):
                    continue
                for m in SHA.finditer(line):
                    leaks.append(f"{name}:{i}: {line.strip()[:100]}")
    assert not leaks, ("the bundle still records a git revision:\n  "
                       + "\n  ".join(sorted(set(leaks))[:10]))


def test_the_bundle_references_nothing_it_withholds():
    """Every import, link and documented command in the bundle must resolve in it.

    Withholding a file is only half the job: `anonymize.py` came out of the bundle
    long before anything checked that nine tests still did `import anonymize`, so
    the first thing a reviewer ran reported nine errors in code that was fine. The
    same trap is set by any README row pointing at a withheld document, or any
    `python3 -m ...` naming a module that does not ship.

    Checked against what the builder would actually write, not against the working
    tree, and phrased as *resolvability* rather than as a list of forbidden names,
    so it keeps holding for whatever gets withheld next.
    """
    import re

    import anonymize

    shipped = {rel.as_posix(): payload
               for rel, payload, _ in anonymize._bundle_entries(anonymize.ROOT)}
    text = {rel: payload.decode("utf-8", "replace") for rel, payload in shipped.items()
            if rel.endswith((".py", ".md", ".sh", ".txt"))}

    # Local modules that exist in the working tree but do not ship. Third-party
    # imports are not the subject: `requirements.txt` covers those.
    def module_paths(name):
        stem = name.replace(".", "/")
        return (f"{stem}.py", f"{stem}/__init__.py")

    withheld_modules = {
        p[:-3].replace("/", ".") for p in
        (str(q.relative_to(anonymize.ROOT).as_posix())
         for q in anonymize.ROOT.rglob("*.py"))
        if p not in shipped and "__pycache__" not in p}

    problems = []
    for rel, body in text.items():
        for line in body.splitlines():
            for name in re.findall(r"^\s*(?:import|from)\s+([\w.]+)", line, re.M):
                root = name.split(".")[0]
                if name in withheld_modules or root in withheld_modules:
                    problems.append(f"{rel}: imports withheld module {name!r}")

        # Markdown only: `[x](args)` is ordinary prose in a docstring, not a link.
        if rel.endswith(".md"):
            for target in re.findall(r"\]\(([^)#\s:]+)(?:#[^)]*)?\)", body):
                if target.startswith(("http", "mailto", "<")):
                    continue
                base = "/".join(rel.split("/")[:-1])
                resolved = _normalize(f"{base}/{target}" if base else target)
                if resolved in shipped or any(k.startswith(resolved.rstrip("/") + "/")
                                              for k in shipped):
                    continue
                problems.append(f"{rel}: links to {target!r}, not in the bundle")

        for mod in re.findall(r"python3? -m ([A-Za-z_][\w.]*)", body):
            # Only modules of *this* tree. `python3 -m venv` and `-m pip` are the
            # interpreter's, and requiring them to ship would be nonsense.
            local = any((anonymize.ROOT / p).exists() for p in module_paths(mod))
            if local and not any(p in shipped for p in module_paths(mod)):
                problems.append(f"{rel}: documents `python -m {mod}`, which is "
                                f"not in the bundle")

    assert not problems, ("the bundle points at files it does not contain:\n  "
                          + "\n  ".join(sorted(set(problems))[:20]))


def _normalize(path: str) -> str:
    """``a/b/../c`` -> ``a/c``, without touching the filesystem."""
    out = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == ".." and out:
            out.pop()
        else:
            out.append(part)
    return "/".join(out)


def test_hardware_record_is_anonymous_and_ascii():
    """The machine record goes into the paper, so it must leak nothing.

    Hostname, username and absolute paths are deliberately not collected: a
    double-blind submission is exactly where "Experiments were run on
    gpu-node-07.lab.university.edu" gets noticed. ASCII-only matters for a
    different reason -- vendor strings carry trademark signs and non-breaking
    spaces, which fail a LaTeX run much later than they are introduced.
    """
    import anonymize
    from common.hardware import as_latex_row, as_sentence, describe

    info = describe("cuda:0")
    rendered = as_sentence(info) + "\n" + as_latex_row("Synthetic quadratic", info)

    assert not anonymize.scan_text("hardware.txt", rendered), (
        "the hardware record trips the anonymity scan:\n  "
        + "\n  ".join(str(f) for f in anonymize.scan_text("hardware.txt", rendered)))
    assert rendered.isascii(), "non-ASCII in the hardware record would break LaTeX"
    for banned in ("hostname", "/home/", "/Users/", "C:" + chr(92) + "Users"):
        assert banned.lower() not in rendered.lower(), f"{banned} leaked into the record"
