"""CPU tests for scientific interpretation and noninterference of EF21 logs."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import torch

from analysis.ef21_report import analyze_rows, schedule_summary, schedule_values
from common.compression_diagnostics import CompressionRecorder, residual_metrics
from common.optimizers import EF21MuonSign
from common.utils import cosine_lr


class ExperimentDiagnosticsTests(unittest.TestCase):
    def test_actual_broadcast_rounding_can_defeat_ideal_alpha(self):
        residual = torch.full((2, 2), 1e-4, dtype=torch.float64)
        # An all-equal residual is ideally reconstructed exactly. Adding its
        # message to a bf16 model near one can round the entire correction away.
        broadcast = torch.ones(2, 2, dtype=torch.bfloat16)
        broadcast.add_(residual.to(torch.bfloat16))
        self.assertTrue(torch.equal(broadcast, torch.ones_like(broadcast)))
        m = residual_metrics(residual, residual, residual, torch.zeros_like(residual),
                             residual, broadcast)
        self.assertAlmostEqual(m["alpha_input"], 1)
        self.assertAlmostEqual(m["observed_q"], 1)
        self.assertAlmostEqual(m["identity_defect"], 1)

    def test_alpha_is_magnitude_concentration_not_sign_coherence(self):
        for value, expected in [(torch.ones(4, 4), 1.0),
                                (torch.eye(4)[0].reshape(2, 2), 0.25),
                                (torch.zeros(4, 4), None)]:
            value = value.double()
            signed = torch.where(value >= 0, 1.0, -1.0)
            post = value-value.abs().mean()*signed
            metrics = residual_metrics(value, value, post, torch.zeros_like(value),
                                       value, torch.zeros_like(value))
            self.assertEqual(metrics["alpha_input"], expected)
            if expected is not None:
                self.assertAlmostEqual(metrics["identity_defect"], 0.0)
                scaled = residual_metrics(7*value, 7*value, 7*post, torch.zeros_like(value),
                                          7*value, torch.zeros_like(value))
                self.assertAlmostEqual(scaled["alpha_input"], expected)

    def test_logging_preserves_nesterov_decay_trajectory_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            def drive(logged):
                torch.manual_seed(31)
                p = torch.nn.Parameter(torch.zeros(4, 5))
                opt = EF21MuonSign([p], lr=0.03, momentum=0.9, nesterov=True,
                                  weight_decay=0.01, decoupled_weight_decay=True,
                                  lmo_dtype=torch.float32, scale_aspect=False)
                if logged:
                    opt.enable_compression_diagnostics(Path(tmp)/"trace.csv")
                for t in range(8):
                    opt.param_groups[0]["lr"] = cosine_lr(0.03, t, 8)
                    p.grad = torch.randn_like(p)
                    opt.step()
                return p.detach().clone(), opt.state[p]["exact_model"].clone(), torch.get_rng_state()
            plain, logged = drive(False), drive(True)
            for a, b in zip(plain, logged):
                self.assertTrue(torch.equal(a, b))
            with (Path(tmp)/"trace.csv").open() as f:
                rows = list(csv.DictReader(f))
            report = analyze_rows(rows)[0]
            self.assertEqual(len(rows), 8)
            self.assertTrue(report["complete_segment"])
            self.assertLess(report["max_envelope_excess"], 1e-12)
            self.assertLess(report["max_energy_closure_relative_error"], 1e-12)
            self.assertFalse(report["stationarity_certificate"])

    def test_missing_rounds_and_legacy_medians_do_not_certify_a_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = CompressionRecorder(Path(tmp)/"trace.csv", every=2)
            x, w = torch.zeros(3, 3), torch.zeros(3, 3)
            for t in range(4):
                rec.begin("probe", x, w, 0.1, 0.1)
                x[0, 0] += 0.1
                residual = x-w
                w += residual.abs().mean()*torch.where(residual >= 0, 1.0, -1.0)
                rec.end("probe", x, w, residual)
            with (Path(tmp)/"trace.csv").open() as f:
                report = analyze_rows(list(csv.DictReader(f)))[0]
            self.assertFalse(report["complete_segment"])
            self.assertIsNone(report["tracking_envelope"])
        legacy = analyze_rows([dict(run_id="old", parameter="c_proj", step="250",
                                    alpha_dn="0.00012", lag_XW="0.1")])[0]
        self.assertEqual(legacy["observation"], "sampled_group_medians")
        self.assertIsNone(legacy["tracking_envelope"])

    def test_schedule_matches_driver_and_filter_impulse_response(self):
        values = schedule_values("cosine", 40, lr=0.2)
        self.assertEqual(values, [cosine_lr(0.2, t, 40) for t in range(40)])
        self.assertGreater(values[-1], 0)  # zero is reached after the last training step
        constant = schedule_summary([0.2]*40)
        self.assertAlmostEqual(constant["smoothness_multiplier"], 0.2)
        self.assertAlmostEqual(constant["effective_steps"], 40)
        for mu in (0, 0.9, 0.99):
            beta = 1-mu
            # Independent frozen-query samples: sum of squared impulse coefficients.
            ema = [beta*mu**k for k in range(10000)]
            nes = [beta*(1+mu)]+[beta*mu**(k+1) for k in range(1, 10000)]
            report = schedule_summary([1.0], mu)
            self.assertAlmostEqual(sum(x*x for x in ema), report["frozen_query_ema_variance_gain"])
            self.assertAlmostEqual(sum(x*x for x in nes), report["frozen_query_nesterov_variance_gain"])
        self.assertRaises(ValueError, schedule_summary, [0.0])

    def test_population_gradient_calibration_for_both_momenta(self):
        from analysis.ef21_calibrate import run_case
        for nes in (False, True):
            for schedule in ("constant", "cosine"):
                result = run_case(steps=8, schedule=schedule, nesterov=nes)
                self.assertLessEqual(result["server_weighted_nuclear_gradient"], result["server_bound"]+1e-10)
                self.assertLessEqual(result["broadcast_weighted_nuclear_gradient"], result["broadcast_bound"]+1e-10)

    def test_federated_recorder_preserves_actual_training(self):
        from tests.test_code import TinyNet, tiny_loader
        from federated.algorithms import run_federated
        with tempfile.TemporaryDirectory() as tmp:
            outputs = []
            for logged in (False, True):
                torch.manual_seed(18)
                model = TinyNet()
                rec = CompressionRecorder(Path(tmp)/"fed.csv", audit_oracle=True) if logged else None
                run_federated("ef21muonsign", model, [tiny_loader()], [tiny_loader()],
                              rounds=3, n_steps=1, lr=0.03, nesterov=True,
                              eval_freq=99, lmo_dtype=torch.float32, verbose=False,
                              compression_recorder=rec)
                outputs.append([v.detach().clone() for v in model.parameters()]+[torch.get_rng_state()])
            for a, b in zip(*outputs):
                self.assertTrue(torch.equal(a, b))
            with (Path(tmp)/"fed.csv").open() as f:
                report = analyze_rows(list(csv.DictReader(f)))
            self.assertTrue(report)
            self.assertTrue(all(r["complete_segment"] for r in report))
            self.assertTrue(all(r["max_direction_operator"] is not None for r in report))


if __name__ == "__main__":
    unittest.main()
