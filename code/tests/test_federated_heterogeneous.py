"""Protocol checks without CIFAR downloads, torchvision, or a GPU."""

import contextlib
import csv
import io
import json
import os
import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from common.paths import results_root
from federated.heterogeneous import (
    DEFAULT_METHODS, HOMOGENEOUS_LRS, PRIORITY_METHODS, REMAINING_METHODS,
    candidates, get_args, job_key, legacy_protocol, protocol, protocol_id, run, select_best,
)


class HeterogeneousExperimentsTests(unittest.TestCase):
    def args(self, *extra):
        return get_args(["--methods", "signmuon", "--rounds", "10",
                         "--seeds", "0", "1", "--tune-seeds", "0", "1",
                         "--lr-points", "5", "--lr-extend-rounds", "0", *extra])

    def fake_runner(self, calls, best=None, fail=None):
        def runner(args, **kw):
            calls.append((args, kw))
            seed, split = kw["seed"], kw["split"]
            self.assertEqual(kw["extra"], ["--partition-seed", str(seed)])
            self.assertEqual((args.partition, args.beta), ("noniid-labeldir", 0.1))
            self.assertEqual(args.momentum, 0.9)
            suffix = f"{kw['tag']}_e{kw['rounds']}_{split[0]}s{seed}"
            log = results_root() / "federated_tuning_logs" / f"{suffix}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"{kw['method']}: synthetic test log", encoding="utf-8")
            if fail and fail(args, kw):
                return None
            path = (results_root() / "federated" /
                    f"{'tune' if split == 'tune' else 'final'}_{suffix}" / f"seed{seed}")
            path.mkdir(parents=True, exist_ok=True)
            # High initial accuracy must not enter the last-five criterion.
            target = HOMOGENEOUS_LRS[kw["method"]] if best is None else best
            val = 83.12345 if kw["lr"] == target else 70.0
            metric = "val_acc" if split == "tune" else "test_acc"
            score = val if split == "tune" else 81.12345 + seed
            data = {"config": {"rounds": kw["rounds"]},
                    "history": {"steps": [0, 2, 4, 6, 8, kw["rounds"]],
                                metric: [99.0] + [score] * 5}}
            (path / "metrics.json").write_text(json.dumps(data), encoding="utf-8")
            (path / "model.pt").write_bytes(b"dummy weights, excluded from archive")
            # Selection must use full-precision saved validation, ignoring both
            # this rounded result and any test-like field supplied by a caller.
            return {"val_acc": 0, "test_acc": 100}
        return runner

    def test_defaults_match_saved_homogeneous_validation_settings(self):
        path = Path(__file__).resolve().parents[1] / "results/federated_overnight/state.json"
        if not path.exists():
            self.skipTest("saved homogeneous results not installed")
        state = json.loads(path.read_text(encoding="utf-8"))
        for method, lr in HOMOGENEOUS_LRS.items():
            self.assertEqual(state["phases"]["lr"][method]["best"]["lr"], lr)
            self.assertEqual([lr for lr, _ in candidates(get_args([]), method)],
                             state["phases"]["lr"][method]["grid"][:5])
        self.assertEqual(state["fingerprint"]["momentum"], 0.9)

    def test_defaults_cover_exactly_the_ten_article_methods(self):
        args = get_args([])
        self.assertEqual(args.methods, DEFAULT_METHODS)
        self.assertEqual(len(args.methods), 10)
        self.assertEqual(set(PRIORITY_METHODS) & set(REMAINING_METHODS), set())
        source = Path(__file__).resolve().parents[2] / "aaai_article/signmuon_body.tex"
        if not source.exists():
            self.skipTest("article source not installed")
        table = source.read_text(encoding="utf-8").split(r"\subsection{Federated Learning}", 1)[1].split(r"\end{table}", 1)[0]
        article_methods = {m.lower().replace("-", "") for m in re.findall(r"^([A-Za-z0-9-]+) &", table, re.M)}
        self.assertEqual(set(args.methods), article_methods)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            get_args(["--momenta", "0.9", "0.99"])

    def test_selection_requires_complete_grid_and_all_tuning_seeds(self):
        args = self.args()
        jobs = {}
        for lr, mu in candidates(args, "signmuon"):
            for seed in args.tune_seeds:
                jobs[job_key("signmuon", lr, mu, seed, "tune")] = {
                    "status": "ok", "val_acc": 80 if lr == 0.05 else 70,
                    "test_acc": 100 if lr == 0.1 else 0}
        missing = jobs.pop(job_key("signmuon", 0.05, 0.9, 1, "tune"))
        self.assertIsNone(select_best(args, jobs, "signmuon"))
        jobs[job_key("signmuon", 0.05, 0.9, 1, "tune")] = missing
        self.assertEqual(select_best(args, jobs, "signmuon"), (0.05, 0.9))

    def test_transfer_tuning_resume_pairing_precision_and_archive(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SIGNMUON_RESULTS": tmp}):
            root = run(self.args("--experiment", "transfer"), self.fake_runner(calls))
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(k["split"] == "full" and a.momentum == 0.9
                                and k["lr"] == 0.1 for a, k in calls))
            run(self.args("--experiment", "tuned"), self.fake_runner(calls))
            # Ten validation trials and the two already-complete finals.
            self.assertEqual(len(calls), 12)
            run(self.args("--experiment", "both"), self.fake_runner(calls))
            self.assertEqual(len(calls), 12)
            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            row = state["jobs"][job_key("signmuon", 0.1, 0.9, 0, "tune")]
            self.assertAlmostEqual(row["val_acc"], 83.12345)
            self.assertEqual(os.environ["SIGNMUON_RESULTS"], tmp)
            self.assertFalse((Path(tmp) / "federated").exists())
            with zipfile.ZipFile(root / "rebuttal_results.zip") as z:
                self.assertFalse(any(x.endswith("model.pt") for x in z.namelist()))
                self.assertEqual(sum(x.endswith("metrics.json") for x in z.namelist()), 12)
                self.assertIn("81.62 +/- 0.71", z.read("SUMMARY.md").decode())
                self.assertIn("transfer", z.read("table.csv").decode())
                self.assertIn("tuned", z.read("table.csv").decode())
            run(self.args("--report-only"), self.fake_runner(calls))
            self.assertEqual(len(calls), 12)

    def test_transfer_is_unsplit_and_tuning_parts_share_cumulative_archive(self):
        calls = []
        options = ["--methods", *DEFAULT_METHODS]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SIGNMUON_RESULTS": tmp}):
            # --tune-part must never reduce transfer to one half of the family.
            root = run(self.args(*options, "--tune-part", "remaining"), self.fake_runner(calls))
            self.assertEqual(len(calls), 20)
            self.assertEqual({kw["method"] for _, kw in calls}, set(DEFAULT_METHODS))
            self.assertTrue(all(kw["split"] == "full" for _, kw in calls))
            self.assertEqual([kw["seed"] for _, kw in calls], [0] * 10 + [1] * 10)

            first = run(self.args(*options, "--experiment", "tuned", "--tune-part", "priority"),
                        self.fake_runner(calls))
            self.assertEqual(first, root)
            self.assertEqual(len(calls), 70)
            self.assertEqual({kw["method"] for _, kw in calls[20:]}, set(PRIORITY_METHODS))
            with zipfile.ZipFile(root / "rebuttal_results.zip") as z:
                rows = list(csv.DictReader(io.StringIO(z.read("table.csv").decode())))
                self.assertEqual(len(rows), 20)
                self.assertTrue(all(r["complete"] == "False" for r in rows
                                    if r["experiment"] == "tuned" and r["tuning_part"] == "remaining"))

            second = run(self.args(*options, "--experiment", "tuned", "--tune-part", "remaining"),
                         self.fake_runner(calls))
            self.assertEqual(second, root)
            self.assertEqual(len(calls), 120)  # All twenty transfer finals reused.
            self.assertEqual({kw["method"] for _, kw in calls[70:]}, set(REMAINING_METHODS))
            state = json.loads((root / "state.json").read_text())
            self.assertEqual(state["tuning_parts"], ["priority", "remaining"])
            with zipfile.ZipFile(root / "rebuttal_results.zip") as z:
                rows = list(csv.DictReader(io.StringIO(z.read("table.csv").decode())))
                self.assertTrue(all(r["complete"] == "True" for r in rows))
                self.assertEqual(sum(x.endswith("metrics.json") for x in z.namelist()), 120)

    def test_tuned_final_uses_validation_winner_and_only_after_grid(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SIGNMUON_RESULTS": tmp}):
            run(self.args("--experiment", "tuned"), self.fake_runner(calls, best=0.05))
        self.assertEqual([k["split"] for _, k in calls], ["tune"] * 10 + ["full"] * 2)
        self.assertTrue(all(a.momentum == 0.9 and k["lr"] == 0.05 for a, k in calls[-2:]))

    def test_failed_trials_are_visible_and_can_be_retried(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SIGNMUON_RESULTS": tmp}):
            args = self.args("--experiment", "tuned")
            with self.assertRaisesRegex(RuntimeError, "some jobs failed"):
                run(args, self.fake_runner(calls, fail=lambda a, k: k["split"] == "tune" and k["lr"] == 0.05))
            root = Path(tmp) / "federated_heterogeneous" / protocol_id(protocol(args))
            self.assertIn("2 failed jobs", (root / "SUMMARY.md").read_text())
            state = json.loads((root / "state.json").read_text())
            with zipfile.ZipFile(root / "rebuttal_results.zip") as z:
                for row in state["jobs"].values():
                    if row["status"] == "failed":
                        self.assertIn("synthetic test log", z.read(row["log"]).decode())
            run(self.args("--experiment", "tuned", "--retry-failed"), self.fake_runner(calls))
            self.assertEqual(len(calls), 14)  # Only the two failed trials are rerun.

    def test_boundary_extension_and_resume_use_the_homogeneous_lattice(self):
        calls = []
        args = self.args("--experiment", "tuned", "--lr-extend-rounds", "3")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SIGNMUON_RESULTS": tmp}):
            root = run(args, self.fake_runner(calls, best=0.2))
            state = json.loads((root / "state.json").read_text())
            self.assertEqual(state["searches"]["signmuon"]["grid"], [.005, .01, .02, .05, .1, .2, .5])
            self.assertEqual(state["searches"]["signmuon"]["extensions"], 1)
            self.assertEqual(len(calls), 16)  # Seven LR values x two seeds + two finals.
            run(args, self.fake_runner(calls, best=0.2))
            self.assertEqual(len(calls), 16)

    def test_original_transfer_cache_is_preserved_when_search_is_corrected(self):
        calls = []
        args = self.args()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SIGNMUON_RESULTS": tmp}):
            old = Path(tmp) / "federated_heterogeneous" / protocol_id(legacy_protocol(protocol(args)))
            with patch("federated.heterogeneous.protocol_directory", return_value=old):
                run(args, self.fake_runner(calls))
            state = json.loads((old / "state.json").read_text())
            state["protocol"] = legacy_protocol(protocol(args))
            state.pop("tuning_protocol", None)
            state.pop("searches", None)
            (old / "state.json").write_text(json.dumps(state), encoding="utf-8")
            self.assertEqual(len(calls), 2)
            root = run(self.args("--experiment", "tuned"), self.fake_runner(calls))
            self.assertEqual(root, old)
            self.assertEqual(len(calls), 12)  # The original two finals were reused.
            updated = json.loads((old / "state.json").read_text())
            self.assertEqual(updated["protocol"]["version"], 3)
            self.assertEqual(updated["previous_protocol"]["version"], 2)

    def test_dry_run_writes_nothing_and_protocol_changes_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SIGNMUON_RESULTS": tmp}):
            run(self.args("--dry-run"), lambda *a, **k: self.fail("dry run launched training"))
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertNotEqual(protocol_id(protocol(self.args())),
                            protocol_id(protocol(self.args("--alpha", "0.5"))))
        self.assertNotEqual(job_key("signmuon", .1, .90000001, 0, "full"),
                            job_key("signmuon", .1, .90000002, 0, "full"))


if __name__ == "__main__":
    unittest.main()
