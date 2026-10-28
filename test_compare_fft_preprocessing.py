"""Run: python -m unittest test_compare_fft_preprocessing -v"""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from compare_fft_preprocessing import compute_conditions, read_gray, run_batch, shared_limits


class FFTPreprocessingTests(unittest.TestCase):
    def test_mean_only_changes_dc_and_hann_matches_definition(self):
        for shape in [(11, 13), (12, 14)]:
            image = np.random.default_rng(4).integers(0, 256, shape, dtype=np.uint8)
            spectra, stats = compute_conditions(image)
            non_dc = np.ones(shape, dtype=bool)
            non_dc[shape[0] // 2, shape[1] // 2] = False
            np.testing.assert_allclose(spectra[0][non_dc], spectra[1][non_dc], atol=1e-10)
            self.assertAlmostEqual(stats[0]["dc_abs"], float(image.sum()))
            self.assertLess(stats[1]["dc_abs"], 1e-9)
            windowed = (image.astype(float) - image.mean()) * np.outer(
                np.hanning(shape[0]), np.hanning(shape[1]))
            expected = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(windowed))))
            np.testing.assert_allclose(spectra[2], expected, atol=1e-10)
            self.assertAlmostEqual(stats[2]["processed_mean"], windowed.mean())
            self.assertAlmostEqual(stats[2]["dc_abs"], abs(windowed.sum()))

    def test_constant_image_and_display_limits(self):
        spectra, stats = compute_conditions(np.full((13, 15), 123, dtype=np.uint16))
        np.testing.assert_array_equal(spectra[1], 0)
        np.testing.assert_array_equal(spectra[2], 0)
        low, high = shared_limits(spectra)
        self.assertTrue(np.isfinite(low) and np.isfinite(high) and high > low)
        full = shared_limits(spectra, 0, 100)
        self.assertAlmostEqual(full[1], np.log1p(123 * 13 * 15))

    def test_batch_continues_after_bad_file_preserves_16bit_and_skips_subfolder(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "input", Path(folder) / "output"
            source.mkdir()
            image = np.arange(120, dtype=np.uint16).reshape(10, 12) * 400
            for name in ["a.TIF", "c.png"]:
                success, encoded = cv2.imencode(Path(name).suffix.lower(), image)
                self.assertTrue(success)
                (source / name).write_bytes(encoded.tobytes())
            (source / "b.jpg").write_bytes(b"invalid image")
            (source / "triming").mkdir()
            (source / "triming" / "ignored.png").write_bytes(b"not an input")
            np.testing.assert_array_equal(read_gray(source / "a.TIF"), image)
            self.assertEqual(read_gray(source / "a.TIF").dtype, np.uint16)
            report = run_batch(source, output)
            self.assertEqual(report["success_count"], 2)
            self.assertEqual(report["error_count"], 1)
            self.assertEqual([Path(r["input"]).name for r in report["successes"]], ["a.TIF", "c.png"])
            self.assertEqual(len(list(output.glob("*_fft_comparison.png"))), 2)
            self.assertEqual(len((output / "statistics.csv").read_text(encoding="utf-8-sig").splitlines()), 7)
            self.assertEqual(json.loads((output / "report.json").read_text())["error_count"], 1)


if __name__ == "__main__":
    unittest.main()
