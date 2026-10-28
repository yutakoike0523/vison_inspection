import unittest
import numpy as np
from block_1d_fft_experiment import analyze
from compare_multiscale_1d_fft import overlap_map


class FFTComparisonTests(unittest.TestCase):
    def test_frequency_amplitude_and_energy(self):
        for n in (128,256):
            x=np.arange(n)
            f,_=analyze(np.tile(100+20*np.cos(2*np.pi*x/64),(n,1)),[.01,.04])
            self.assertAlmostEqual(f['x_plain_peak_frequency'],1/64)
            self.assertAlmostEqual(f['x_plain_peak_amplitude'],20)
            self.assertAlmostEqual(f['x_plain_total_energy'],200)
            self.assertAlmostEqual(f['x_plain_band_energy_ratio'],1)
            self.assertLess(abs(f['x_hann_peak_amplitude']-20),.03)
            self.assertFalse(f['y_hann_spectrum_defined'])

    def test_constant_and_nyquist(self):
        for n in (128,256):
            f,_=analyze(np.ones((n,n))*100,[.01,.04])
            self.assertIsNone(f['x_hann_band_energy_ratio'])
            f,_=analyze(np.tile(100+10*(-1.)**np.arange(n),(n,1)),[.01,.04])
            self.assertAlmostEqual(f['x_plain_peak_amplitude'],10)
            self.assertAlmostEqual(f['x_plain_total_energy'],100)

    def test_overlap_missing_and_uncovered(self):
        rows=[dict(x_start=0,y_start=0,block_size=2,v=2.),dict(x_start=1,y_start=0,block_size=2,v=4.),
              dict(x_start=0,y_start=1,block_size=2,v=None)]
        m,c=overlap_map((4,4),rows,'v')
        np.testing.assert_allclose(m[:2,:3],[[2,3,4],[2,3,4]])
        self.assertTrue(np.isnan(m[2:]).all())
        self.assertEqual(c[0,1],2)

    def test_same_center(self):
        cx,cy=1088,576
        for n in (128,256):
            x,y=cx-n//2,cy-n//2
            self.assertEqual(x+n/2,cx)
            self.assertEqual(y+n/2,cy)


if __name__=='__main__': unittest.main()
