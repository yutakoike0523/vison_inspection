"""Numerical behavior checks using known carrier phases, not labeled real defects."""
import unittest
from dataclasses import replace

import numpy as np

from phase_anomaly_detection import Config, analyze, angle_distance


class FourierBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape = (384, 512)
        y, x = np.indices(cls.shape)
        cls.bump = np.exp(-((x-270)**2 + (y-200)**2) / (2*23**2))
        cls.phase1 = 2*np.pi*(13*x/512 + 8*y/384)
        cls.phase2 = 2*np.pi*(-9*x/512 + 14*y/384)
        cls.manual = [(13, 8), (-9, 14)]
        cls.config = Config(band_sigma=3, phase_sigma=48, amplitude_sigma=48)
        cls.center = cls.bump > .5
        cls.controls = (cls.bump < .01) & (x > 130) & (x < 390) & (y > 100) & (y < 290)
        cls.clean = analyze(cls.make_image(), cls.config, cls.manual)

    @classmethod
    def make_image(cls, phase_bump=0, amplitude_drop=0, curvature=0):
        y, x = np.indices(cls.shape)
        smooth = curvature*((x-256)/256)**2
        return (120 + 30*(1-amplitude_drop*cls.bump)*np.cos(cls.phase1 + smooth + phase_bump*cls.bump)
                + 30*np.cos(cls.phase2 + .6*smooth - .7*phase_bump*cls.bump))

    def test_known_carriers_automatic_selection(self):
        result = analyze(self.make_image(), self.config)
        found = {(p["dx"], p["dy"]) for p in result["peaks"]}
        self.assertEqual(found, set(self.manual))
        self.assertGreater(angle_distance(*result["peaks"]), 35)

    def test_single_sideband_and_true_complex_phase(self):
        first, second = self.clean["directions"]
        self.assertEqual(np.count_nonzero(first["mask"]*second["mask"]), 0)
        for p, d in zip(self.clean["peaks"], self.clean["directions"]):
            self.assertEqual(d["mask"][192, 256], 0)
            self.assertEqual(d["mask"][p["symmetric_y"], p["symmetric_x"]], 0)
            valid = d["phase_valid"]
            error = abs(np.exp(1j*d["unwrapped"][valid]) - np.exp(1j*d["wrapped"][valid]))
            self.assertLess(float(error.max()), 1e-10)

    def test_clean_carrier_has_no_hann_anomaly(self):
        self.assertLess(float(np.nanpercentile(self.clean["phase_score"], 95)), .01)
        self.assertLess(float(np.nanpercentile(self.clean["amplitude_score"], 95)), .01)
        self.assertAlmostEqual(float(np.median(self.clean["directions"][0]["amplitude_corrected"][self.center])), 15, delta=.1)

    def test_local_phase_perturbation_above_smooth_curvature(self):
        smooth = analyze(self.make_image(curvature=.5), self.config, self.manual)
        defect = analyze(self.make_image(phase_bump=1.3, curvature=.5), self.config, self.manual)
        response = float(np.nanmedian(defect["phase_score"][self.center]))
        reference = float(np.nanmedian(smooth["phase_score"][self.center]))
        elsewhere = float(np.nanmedian(defect["phase_score"][self.controls]))
        self.assertGreater(response, 5*reference)
        self.assertGreater(response, 3*elsewhere)
        self.assertGreater(response, .05)

    def test_local_amplitude_loss_is_detected(self):
        defect = analyze(self.make_image(amplitude_drop=.75), self.config, self.manual)
        response = float(np.nanmedian(defect["amplitude_score"][self.center]))
        elsewhere = float(np.nanmedian(defect["amplitude_score"][self.controls]))
        self.assertGreater(response, 3*elsewhere)
        self.assertGreater(response, .05)
        # A pure amplitude modulation should not turn into a large phase defect.
        self.assertLess(float(np.nanmedian(defect["phase_score"][self.center])), .03)

    def test_odd_rectangular_image_and_invalid_edges(self):
        result = analyze(self.make_image()[:-1, :-1], self.config)
        self.assertTrue(all(p["symmetry_ratio"] > .999999 for p in result["peaks"]))
        self.assertFalse(result["phase_valid"][0].any())
        self.assertTrue(np.isnan(result["total_score"][0]).all())

    def test_geometric_roi_excludes_outside_values(self):
        roi = np.ones(self.shape, bool)
        roi[:, :60] = False
        a = self.make_image()
        b = a.copy()
        b[~roi] = 255
        first = analyze(a, self.config, self.manual, roi)
        second = analyze(b, self.config, self.manual, roi)
        np.testing.assert_allclose(first["spectrum"], second["spectrum"], atol=1e-9)
        self.assertFalse(first["phase_valid"][~roi].any())

    def test_full_area_mode_keeps_edges_and_low_amplitude_pixels(self):
        result = analyze(self.make_image(), self.config, self.manual, full_area=True)
        self.assertTrue(result["phase_valid"].all())
        self.assertTrue(result["amplitude_valid"].all())
        self.assertTrue(np.isfinite(result["total_score"]).all())
        self.assertTrue(np.isfinite(result["normalized_phase_score"]).all())
        no_hann = analyze(self.make_image(), self.config, self.manual,
                          full_area=True, use_hann=False)
        np.testing.assert_array_equal(no_hann["hann"], np.ones(self.shape))

    def test_bad_inputs_fail_explicitly(self):
        for config in [replace(self.config, band_sigma=0), replace(self.config, phase_sigma=float("nan"))]:
            with self.assertRaises(ValueError):
                analyze(self.make_image(), config)
        with self.assertRaises(ValueError):
            analyze(np.full(self.shape, 120), self.config)
        with self.assertRaises(ValueError):
            analyze(self.make_image(), self.config, [(13, 8), (-13, -8)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
