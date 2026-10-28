"""Numerical and synthetic-signal checks; not validation on real defects."""
import unittest

import numpy as np

from fft_residual_detector import build_mask, manual_peak_mask, normalized_power, residual_score


def synthetic_grid(seed, *, scratch=False, offset=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:256, :256]
    phase1, phase2 = rng.uniform(0, 2 * np.pi, 2)
    grid = (128 + offset + 25 * np.cos(2 * np.pi * (10*x + 10*y) / 256 + phase1)
            + 25 * np.cos(2 * np.pi * (10*x - 10*y) / 256 + phase2)
            + rng.normal(0, 1.5, x.shape))
    if scratch:
        # A bright horizontal scratch with tapered ends, crossing the texture.
        grid += 45 * np.exp(-0.5 * ((y - 123) / 3) ** 2) * np.exp(-0.5 * ((x - 128) / 70) ** 2)
    return np.clip(np.rint(grid), 0, 255).astype(np.uint8)


class ResidualDetectorTests(unittest.TestCase):
    def test_dc_and_parseval_for_odd_and_even_shapes(self):
        rng = np.random.default_rng(7)
        for shape in ((63, 65), (64, 64)):
            gray = rng.uniform(0, 255, shape)
            win = np.outer(np.hanning(shape[0]), np.hanning(shape[1]))
            mean = (gray * win).sum() / win.sum()
            expected = (((gray - mean) * win) ** 2).sum() / (win ** 2).sum()
            power = normalized_power(gray)
            self.assertAlmostEqual(power.sum(), expected, places=9)
            self.assertLess(power[shape[0] // 2, shape[1] // 2], 1e-20)
            np.testing.assert_allclose(normalized_power(gray + 20), power, atol=1e-12)

    def test_normal_peaks_and_added_scratch(self):
        reference = np.mean([normalized_power(synthetic_grid(i)) for i in range(4)], axis=0)
        keep, peak_mask, meta = build_mask(reference)
        self.assertGreater(len(meta["peak_centers_cycles_per_pixel"]), 0)
        for y, x in ((138, 138), (118, 118), (118, 138), (138, 118)):
            self.assertTrue(peak_mask[y, x])
            self.assertFalse(keep[y, x])
        # Calibration images are separate from mask training and test images.
        threshold = max(residual_score(normalized_power(synthetic_grid(i)), keep)
                        for i in range(10, 20)) * 1.1
        normal = residual_score(normalized_power(synthetic_grid(40)), keep)
        scratch = residual_score(normalized_power(synthetic_grid(40, scratch=True)), keep)
        self.assertLess(normal, threshold)
        self.assertGreater(scratch, threshold)
        self.assertGreater(scratch, 5 * normal)
        # Normal periodic peaks carry most of the texture energy and are removed.
        self.assertLess(normal, normalized_power(synthetic_grid(40)).sum() * 0.01)

    def test_symmetry(self):
        for shape in ((63, 65), (64, 64)):
            y, x = np.indices(shape)
            reference = normalized_power(128 + 20 * np.cos(2*np.pi*(3*x/shape[1]+3*y/shape[0])))
            keep, peak_mask, _ = build_mask(reference, low=0.2)
            for mask in (keep, peak_mask):
                unshifted = np.fft.ifftshift(mask)
                opposite = unshifted[np.ix_((-np.arange(shape[0])) % shape[0],
                                           (-np.arange(shape[1])) % shape[1])]
                np.testing.assert_array_equal(unshifted, opposite)

    def test_bad_settings_and_size(self):
        reference = normalized_power(synthetic_grid(0))
        for settings in ({"low": 0.001}, {"dc": 0.1}, {"peak_radius": -1},
                         {"low": float("nan")}, {"peak_relative": 0}):
            with self.assertRaises(ValueError):
                build_mask(reference, **settings)
        with self.assertRaises(ValueError):
            residual_score(reference, np.ones((32, 32), dtype=bool))

    def test_user_coordinates_are_fixed_and_symmetric(self):
        for shape in ((63, 65), (64, 64)):
            keep, masked, meta = manual_peak_mask(shape, [(28, 25)], low=0.2)
            self.assertTrue(masked[25, 28])
            cx, cy = shape[1] // 2, shape[0] // 2
            self.assertTrue(masked[2*cy-25, 2*cx-28])
            self.assertFalse(keep[cy, cx])
            self.assertEqual(meta['requested_peaks_xy'], [[28, 25]])
            for mask in (keep, masked):
                unshifted = np.fft.ifftshift(mask)
                opposite = unshifted[np.ix_((-np.arange(shape[0])) % shape[0],
                                           (-np.arange(shape[1])) % shape[1])]
                np.testing.assert_array_equal(unshifted, opposite)
        with self.assertRaises(ValueError):
            manual_peak_mask((64, 64), [(64, 0)])

    def test_manual_mask_preserves_added_scratch(self):
        keep, _, _ = manual_peak_mask((256, 256), [(138, 138), (138, 118)], low=0.08)
        normal = residual_score(normalized_power(synthetic_grid(40)), keep)
        scratch = residual_score(normalized_power(synthetic_grid(40, scratch=True)), keep)
        self.assertGreater(scratch, 5 * normal)


if __name__ == "__main__":
    unittest.main()
