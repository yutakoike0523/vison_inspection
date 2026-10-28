"""Numerical and scoring checks for the local FFT experiment."""

import unittest

import numpy as np

from local_fft_block_experiment import (
    WEIGHTS, analyze_blocks, axial_distance, axial_median, block_features,
    block_positions, build_reference, rasterize_scores, score_records, rank_blocks,
)


class LocalFFTTests(unittest.TestCase):
    @staticmethod
    def grid(shape=(512, 512), period=16):
        y, x = np.mgrid[:shape[0], :shape[1]]
        return 120 + 30 * np.cos(2 * np.pi * x / period) + 20 * np.cos(2 * np.pi * y / period)

    def test_grid_peak_period_energy_and_no_input_mutation(self):
        for size in (128, 256):
            block = self.grid((size, size))
            original = block.copy()
            features, fft = block_features(block)
            np.testing.assert_array_equal(block, original)
            self.assertAlmostEqual(features["main_peak_period_pixels"], 16)
            self.assertTrue(features["symmetric_peak_found"])
            self.assertGreater(features["peak_energy_ratio"], .99)
            self.assertLessEqual(features["peak_energy_ratio"], 1)
            self.assertAlmostEqual(features["peak_strength_normalized"], 15, delta=.01)
            self.assertGreater(features["spectral_spread"], 0)
            # Display exclusion has not erased the raw center spectrum.
            self.assertEqual(fft["spectrum"].shape, (size, size))

    def test_corrupted_block_scores_higher_with_both_reference_modes(self):
        image = self.grid()
        image[128:256, 256:384] += np.random.default_rng(42).normal(0, 30, (128, 128))
        for mode in ("median", "manual"):
            rows, reference = analyze_blocks(image, "synthetic", 128, 128, mode,
                                             [(0, 0), (0, 1)], WEIGHTS)
            defect = next(r for r in rows if (r["block_row"], r["block_col"]) == (1, 2))
            normal = next(r for r in rows if (r["block_row"], r["block_col"]) == (0, 0))
            self.assertGreater(defect["anomaly_score"], normal["anomaly_score"] + 1)
            self.assertLess(defect["peak_energy_ratio"], normal["peak_energy_ratio"])
            self.assertGreater(defect["spectral_spread"], normal["spectral_spread"])
            self.assertTrue(reference["valid"])
            self.assertEqual(defect["anomaly_score"], sum(defect[k] * w for k, w in WEIGHTS.items()))

    def test_edges_and_overlap_averaging(self):
        positions = list(block_positions((301, 421), 128, 128))
        self.assertEqual(len(positions), 6)
        self.assertEqual(positions[-1], (1, 2, 256, 128))
        rows = [dict(x_start=0, y_start=0, block_size=4, anomaly_score=1.),
                dict(x_start=2, y_start=0, block_size=4, anomaly_score=3.)]
        heatmap = rasterize_scores((5, 7), rows)
        np.testing.assert_array_equal(heatmap[:4, :2], 1.)
        np.testing.assert_array_equal(heatmap[:4, 2:4], 2.)
        np.testing.assert_array_equal(heatmap[:4, 4:6], 3.)
        self.assertTrue(np.isnan(heatmap[4, :]).all())
        self.assertTrue(np.isnan(heatmap[:, 6]).all())

    def test_angle_wrap_and_invalid_manual_reference(self):
        self.assertEqual(float(axial_distance(179, 1)), 2)
        self.assertLessEqual(float(axial_distance(axial_median([179, 1, 0]), 0)), 1)
        rows, _ = analyze_blocks(self.grid((256, 256)), "grid", 128, 128)
        with self.assertRaises(ValueError):
            build_reference(rows, "manual", [(20, 20)])
        with self.assertRaises(ValueError):
            build_reference(rows, "manual", [])

    def test_missing_peaks_and_degenerate_reference_are_explicit(self):
        image = self.grid((256, 256))
        image[:128, 128:] = 120
        rows, _ = analyze_blocks(image, "grid", 128, 128, "manual", [(1, 0)])
        missing = next(r for r in rows if (r["block_row"], r["block_col"]) == (0, 1))
        self.assertEqual(missing["feature_status"], "no_peak_candidates")
        self.assertEqual(missing["anomaly_score"], sum(WEIGHTS.values()))
        blank, reference = analyze_blocks(np.full((256, 256), 120), "flat", 128, 128)
        self.assertFalse(reference["valid"])
        self.assertTrue(all(r["anomaly_score"] is None for r in blank))

    def test_manual_reference_and_weights_control_score(self):
        rows, reference = analyze_blocks(self.grid((256, 256)), "grid", 128, 128, "manual", [(0, 0)])
        self.assertEqual(reference["reference_count"], 1)
        modified = dict(rows[1])
        modified["main_peak_angle"] = (reference["main_peak_angle"] + 90) % 180
        weights = {name: (2. if name == "peak_angle_error" else 0.) for name in WEIGHTS}
        score_records([modified], reference, weights)
        self.assertEqual(modified["anomaly_score"], 2.)

    def test_ratio_map_uses_raw_ratio_without_reference_and_ranks_low_first(self):
        rows = [dict(x_start=0, y_start=0, block_size=4, anomaly_score=None,
                     peak_energy_ratio=.2, peak_energy_ratio_defined=True),
                dict(x_start=2, y_start=0, block_size=4, anomaly_score=4.,
                     peak_energy_ratio=.8, peak_energy_ratio_defined=True),
                dict(x_start=6, y_start=0, block_size=4, anomaly_score=5.,
                     peak_energy_ratio=0., peak_energy_ratio_defined=False)]
        heatmap = rasterize_scores((4, 10), rows, "peak_energy_ratio")
        np.testing.assert_allclose(heatmap[:, :2], .2)
        np.testing.assert_allclose(heatmap[:, 2:4], .5)
        np.testing.assert_allclose(heatmap[:, 4:6], .8)
        self.assertTrue(np.isnan(heatmap[:, 6:]).all())
        ranked = rank_blocks(rows, "peak_energy_ratio")
        self.assertEqual([r["peak_energy_ratio"] for r in ranked], [.2, .8])
        self.assertEqual([r["anomaly_score"] for r in rank_blocks(rows, "anomaly_score")], [5., 4.])

    def test_zero_energy_ratio_is_undefined_and_not_colored_red(self):
        rows, _ = analyze_blocks(np.full((128, 128), 120), "flat", 128, 128)
        self.assertFalse(rows[0]["peak_energy_ratio_defined"])
        self.assertTrue(np.isnan(rasterize_scores((128, 128), rows, "peak_energy_ratio")).all())
        self.assertEqual(rank_blocks(rows, "peak_energy_ratio"), [])


if __name__ == "__main__":
    unittest.main()
