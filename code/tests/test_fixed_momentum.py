"""Finite stochastic-law checks of the closed fixed-radius EF21 bound."""

import unittest

from analysis.ef21_fixed_calibrate import enumerate_case


class FixedMomentumTests(unittest.TestCase):
    def test_all_noise_histories_both_momenta_downlinks_and_schedules(self):
        for nes in (False, True):
            for down in ("exact", "sign"):
                for schedule in ("constant", "cosine"):
                    result = enumerate_case(steps=4, nesterov=nes, downlink=down, schedule=schedule)
                    self.assertEqual(result["enumerated_paths"], 16)
                    self.assertTrue(result["checked"])

    def test_independent_workers_at_adaptive_queries(self):
        for nes in (False, True):
            result = enumerate_case(steps=3, workers=2, nesterov=nes)
            self.assertEqual(result["enumerated_paths"], 64)
            self.assertTrue(result["checked"])

    def test_biased_roundoff_in_momentum_targets_and_compression(self):
        for nes in (False, True):
            result = enumerate_case(steps=4, nesterov=nes, roundoff=1e-4)
            self.assertEqual(result["enumerated_paths"], 16)
            self.assertTrue(result["checked"])


if __name__ == "__main__":
    unittest.main()
