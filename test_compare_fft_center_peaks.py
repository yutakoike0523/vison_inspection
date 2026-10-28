"""Run: python -m unittest test_compare_fft_center_peaks -v"""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from compare_fft_center_peaks import center_zoom, detect_peaks, display_limits, hann_fft, run_batch


class CenterPeakTests(unittest.TestCase):
    def test_fft_matches_requested_operations_and_preserves_input(self):
        gray = np.random.default_rng(3).integers(0, 255, (57, 80), dtype=np.uint8)
        original = gray.copy()
        transformed, spectrum = hann_fft(gray)
        expected = np.fft.fftshift(np.fft.fft2((gray.astype(float) - gray.mean()) *
                                   np.outer(np.hanning(57), np.hanning(80))))
        np.testing.assert_allclose(transformed, expected)
        np.testing.assert_allclose(spectrum, np.log1p(np.abs(expected)))
        np.testing.assert_array_equal(gray, original)

    def test_known_periodic_peaks_on_odd_and_even_rectangular_images(self):
        for height, width in [(191, 257), (192, 256)]:
            yy, xx = np.mgrid[:height, :width]
            gray = (120 + 30 * np.cos(2 * np.pi * (18 * xx / width + 8 * yy / height))
                    + 20 * np.cos(2 * np.pi * (-11 * xx / width + 22 * yy / height)))
            transformed, spectrum = hann_fft(gray)
            zoom, valid, bounds, _ = center_zoom(spectrum, 60, 5)
            peaks, threshold = detect_peaks(zoom, valid, bounds, gray.shape, max_peaks=4)
            found = {(p["x_offset"], p["y_offset"]) for p in peaks}
            self.assertEqual(found, {(18, 8), (-18, -8), (-11, 22), (11, -22)})
            for peak in peaks:
                expected_period = 1 / np.hypot(peak["x_offset"] / width, peak["y_offset"] / height)
                self.assertAlmostEqual(peak["period_pixels"], expected_period)
                self.assertGreaterEqual(peak["log_amplitude"], threshold)
            for first, second in zip(peaks[::2], peaks[1::2]):
                self.assertEqual(first["pair_id"], second["pair_id"])
                self.assertEqual(first["x_offset"], -second["x_offset"])
                self.assertEqual(first["y_offset"], -second["y_offset"])

    def test_small_zoom_bounds_and_display_mask_do_not_change_fft(self):
        for shape in [(7, 12), (8, 13)]:
            spectrum = np.arange(np.prod(shape), dtype=float).reshape(shape)
            original = spectrum.copy()
            zoom, valid, bounds, extent = center_zoom(spectrum, 150, 2)
            self.assertEqual(zoom.shape, shape)
            self.assertEqual(bounds, (0, shape[1], 0, shape[0]))
            self.assertFalse(valid[shape[0] // 2, shape[1] // 2])
            low, high = display_limits(zoom[valid])
            masked = zoom.copy()
            masked[~valid] = low
            np.testing.assert_array_equal(spectrum, original)
            np.testing.assert_array_equal(zoom, original)
            self.assertGreater(high, low)

    def test_constant_image_and_completely_masked_zoom_have_no_peaks(self):
        _, spectrum = hann_fft(np.full((15, 19), 125, dtype=np.uint16))
        for radius in (2, 100):
            zoom, valid, bounds, _ = center_zoom(spectrum, 150, radius)
            peaks, threshold = detect_peaks(zoom, valid, bounds, spectrum.shape)
            self.assertEqual(peaks, [])
            self.assertIsNone(threshold)

    def test_mask_does_not_create_peak_on_monotonic_dc_tail(self):
        yy, xx = np.mgrid[-30:31, -30:31]
        spectrum = 20 - np.hypot(xx, yy) / 10
        zoom, valid, bounds, _ = center_zoom(spectrum, 30, 10)
        peaks, _ = detect_peaks(zoom, valid, bounds, spectrum.shape)
        self.assertEqual(peaks, [])

    def test_batch_records_failures_and_continues_nonrecursively(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "input", Path(folder) / "output"
            source.mkdir()
            (source / "a.jpg").write_bytes(b"bad image")
            gray = np.random.default_rng(4).integers(0, 65535, (29, 35), dtype=np.uint16)
            _, encoded = cv2.imencode(".png", gray)
            (source / "b.PNG").write_bytes(encoded.tobytes())
            (source / "triming").mkdir()
            (source / "triming" / "ignored.png").write_bytes(b"bad")
            report = run_batch(source, output, max_peaks=5)
            self.assertEqual(report["error_count"], 1)
            self.assertEqual(report["success_count"], 1)
            record = json.loads((output / "b.PNG_peaks.json").read_text())
            self.assertEqual(record["width"], 35)
            self.assertEqual(record["height"], 29)
            self.assertLessEqual(record["detected_peak_count"], 4)
            self.assertTrue((output / "b.PNG_fft_center_comparison.png").exists())
            self.assertTrue((output / "peaks.csv").exists())


if __name__ == "__main__":
    unittest.main()
