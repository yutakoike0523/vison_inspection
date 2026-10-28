"""Numerical and callback checks: python -m unittest test_interactive_2d_fft"""

from types import SimpleNamespace
import tkinter as tk
import unittest

import numpy as np

from interactive_2d_fft import FFTViewer, FrequencySelection, SOURCE


class ReconstructionTests(unittest.TestCase):
    def test_every_pair_matches_masked_ifft_and_sum_recovers_input(self):
        for shape in [(5, 7), (6, 8), (5, 8), (6, 7)]:
            gray = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8)
            model = FrequencySelection(gray)
            for y in range(shape[0]):
                for x in range(shape[1]):
                    key = model.key(x, y)
                    ky, kx = key
                    mask = np.zeros(shape, dtype=complex)
                    mask[ky, kx] = model.fft[ky, kx]
                    partner = ((-ky) % shape[0], (-kx) % shape[1])
                    mask[partner] = model.fft[partner]
                    np.testing.assert_allclose(model.component(key), np.fft.ifft2(mask).real, atol=1e-10)
                    model.add(key)
            np.testing.assert_allclose(model.total, gray, atol=1e-10)
            self.assertFalse(model.add(model.selected[0]))
            while model.selected:
                model.undo()
            np.testing.assert_array_equal(model.total, 0)

    def test_dc_retains_mean_and_clear_resets(self):
        model = FrequencySelection(np.full((6, 8), 125, dtype=np.uint8))
        dc = model.key(4, 3)
        model.add(dc)
        np.testing.assert_allclose(model.total, 125)
        model.clear()
        self.assertEqual(model.selected, [])
        np.testing.assert_array_equal(model.total, 0)


class ViewerTests(unittest.TestCase):
    def test_hover_click_duplicate_undo_and_image_switch(self):
        root = tk.Tk()
        root.withdraw()
        paths = [SOURCE / "image18.png", SOURCE / "triming" / "image18_triming.png"]
        viewer = FFTViewer(root, paths)
        viewer.result.withdraw()
        try:
            event = SimpleNamespace(inaxes=viewer.fft_ax, xdata=230., ydata=160., button=1)
            viewer.on_motion(event)
            root.after_cancel(viewer.timer)
            viewer.apply_hover()
            self.assertIsNotNone(viewer.hover_wave)
            viewer.on_click(event)
            self.assertEqual(len(viewer.model.selected), 1)
            np.testing.assert_allclose(viewer.model.total, viewer.hover_wave)
            # Clicking the conjugate partner must not add it twice.
            partner = viewer.model.points(viewer.hover_key)[0]
            event.xdata, event.ydata = partner
            viewer.on_click(event)
            self.assertEqual(len(viewer.model.selected), 1)
            event.xdata += 2
            viewer.on_click(event)
            self.assertEqual(len(viewer.model.selected), 2)
            np.testing.assert_allclose(viewer.result_artists[2].get_array(), viewer.model.total)
            viewer.undo()
            self.assertEqual(len(viewer.model.selected), 1)
            viewer.clear()
            np.testing.assert_array_equal(viewer.model.total, 0)
            viewer.toggle_kind()
            self.assertEqual(viewer.index, 1)
            self.assertIsNone(viewer.hover_key)
            self.assertEqual(viewer.model.gray.shape, (294, 281))
            viewer.canvas.draw()
            viewer.result_canvas.draw()
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
