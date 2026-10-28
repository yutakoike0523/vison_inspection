import unittest
from types import SimpleNamespace
import numpy as np
from fft2_cosine_similarity import fft_amplitude,common_spectrum,comparison_mask,vector,cosine,percentile,area_weights,block_peak_matches


def wave(h,w,fx,fy=0):
    y,x=np.mgrid[:h,:w]
    return 100+20*np.cos(2*np.pi*(fx*x+fy*y))


class CosineTests(unittest.TestCase):
    def test_frequency_coordinates_rectangular_global(self):
        _,a=fft_amplitude(wave(512,768,1/32,1/64))
        c=common_spectrum(a)
        y,x=np.unravel_index(np.argmax(c),c.shape)
        f=np.linspace(-.125,.125,129)
        self.assertAlmostEqual(abs(f[x]),1/32)
        self.assertAlmostEqual(abs(f[y]),1/64)
        m=comparison_mask(129,5)
        for n in (128,256):
            _,b=fft_amplitude(wave(n,n,1/32,1/64))
            _,other=fft_amplitude(wave(n,n,0,1/16))
            good=cosine(vector(c,m,'linear'),vector(common_spectrum(b),m,'linear'))
            bad=cosine(vector(c,m,'linear'),vector(common_spectrum(other),m,'linear'))
            self.assertGreater(good,bad+.2)

    def test_brightness_and_identity(self):
        a=wave(128,128,1/32)
        _,f=fft_amplitude(a); _,g=fft_amplitude(a*3+70)
        m=comparison_mask(129,5)
        v=vector(common_spectrum(f),m,'linear'); u=vector(common_spectrum(g),m,'linear')
        self.assertAlmostEqual(cosine(v,u),1)
        self.assertAlmostEqual(cosine(v,v),1)
        self.assertAlmostEqual(cosine(vector(common_spectrum(f),m,'log'),vector(common_spectrum(f),m,'log')),1)

    def test_constant_undefined(self):
        _,f=fft_amplitude(np.ones((128,128))*100)
        m=comparison_mask(129,5); v=vector(common_spectrum(f),m,'linear')
        self.assertIsNone(cosine(v,v))

    def test_area_weights_and_mask(self):
        for n in (128,256,1536,2048):
            np.testing.assert_allclose(np.asarray(area_weights(n,129,.125).sum(1)).ravel(),1)
        np.testing.assert_allclose(common_spectrum(np.ones((128,256))),1)
        m=comparison_mask(129,5)
        self.assertEqual(np.sum(~m),121)
        self.assertFalse(m[64,69]); self.assertTrue(m[64,70])

    def test_percentiles_and_ties(self):
        values=np.array([.1,.2,.2,.4])
        self.assertEqual(percentile(values,.2),50)
        self.assertEqual(percentile(values,.1),12.5)
        self.assertEqual(percentile(values,.3),75)
        self.assertIsNone(percentile(values,None))

    def test_peak_match_uses_frequency_not_full_image_bin(self):
        # A full-image reference at 1/16 cycles/pixel is expected at +8 bins in
        # a 128px block, not at the full image coordinate (for example +48).
        args=SimpleNamespace(peak_dc_radius=3,peak_percentile=95.,peak_local_max_size=3,peak_tolerance=1.)
        image=wave(128,128,1/16)
        raw=np.abs(np.fft.fftshift(np.fft.fft2(image-image.mean())))
        refs=[dict(peak_id=1,pair_id=1,fx=1/16,fy=0),dict(peak_id=2,pair_id=1,fx=-1/16,fy=0)]
        matches,_=block_peak_matches(raw,refs,args)
        self.assertTrue(all(m['matched'] for m in matches))
        self.assertEqual({m['block_dx'] for m in matches},{-8,8})


if __name__=='__main__': unittest.main()
