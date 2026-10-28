"""Small deterministic checks for conservative exclusion and fractional GT retention."""
import unittest
import numpy as np
import point_anomaly_fft_mask as m


class MaskTests(unittest.TestCase):
    def rows(self):
        return [dict(size=n,x=x,y=y,similarity=.2,peak_strength=1.,valid_peak=True)
                for n in m.SIZES for y in m.windows(384,n) for x in m.windows(384,n)]

    def test_both_scales_required(self):
        rows=self.rows()
        self.assertTrue(m.conservative_mask((384,384),rows,.7,2).all())
        self.assertFalse(m.conservative_mask((384,384),[r for r in rows if r['size']==128],.7,2).any())

    def test_unknown_or_strong_peak_veto(self):
        for change in [dict(valid_peak=False),dict(peak_strength=7.),dict(similarity=None),dict(similarity=.7)]:
            rows=self.rows();rows[0].update(change)
            self.assertFalse(m.conservative_mask((384,384),rows,.7,2)[:128,:128].any())

    def test_uncovered_strip_is_retained(self):
        rows=[dict(size=n,x=x,y=y,similarity=.2,peak_strength=1.,valid_peak=True)
              for n in m.SIZES for y in m.windows(390,n) for x in m.windows(390,n)]
        excluded=m.conservative_mask((390,390),rows,.7,2)
        self.assertFalse(excluded[384:,:].any())
        self.assertFalse(excluded[:,384:].any())

    def test_fractional_area(self):
        excluded=np.zeros((4,4),bool)
        excluded[0,0]=True
        center,area=m.retention([.5,.5,2.5,2.5],excluded)
        self.assertTrue(center)
        self.assertAlmostEqual(area,1-.25/4)

    def test_grid_precedence_over_percentile(self):
        rows=[dict(size=256,x=0,y=0,similarity=.95),dict(size=256,x=128,y=0,similarity=.96)]
        self.assertFalse(m.legacy_mask((256,384),rows).any())

    def test_grid_wins_equal_vote(self):
        rows=[dict(size=256,x=0,y=0,similarity=.1),dict(size=256,x=128,y=0,similarity=.95)]
        ex=m.legacy_mask((256,384),rows)
        self.assertTrue(ex[:,:128].all())
        self.assertFalse(ex[:,128:].any())

    def test_one_direction_is_not_stable(self):
        self.assertFalse(m.two_directions([dict(pair_id=1,fx=.1,fy=0),dict(pair_id=2,fx=.2,fy=0)]))
        self.assertTrue(m.two_directions([dict(pair_id=1,fx=.1,fy=0),dict(pair_id=2,fx=0,fy=.1)]))


if __name__=='__main__':unittest.main()
