"""Deterministic checks for the saved principle-analysis artifacts."""
import csv
import json
import unittest
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/fft_region_analysis'


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


class AnalysisArtifactTests(unittest.TestCase):
    def test_roi_definitions_are_in_bounds_and_reviewed(self):
        data=json.loads((OUT/'roi_definitions.json').read_text(encoding='utf-8'))
        self.assertEqual(len({r['id'] for r in data['regions']}),len(data['regions']))
        for r in data['regions']:
            w,h=(428,321) if r['image']=='image18' else (2048,1536)
            x1,y1,x2,y2=r['box']
            self.assertTrue(0<=x1<x2<=w and 0<=y1<y2<=h)
            self.assertEqual(x2-x1,r['size']);self.assertEqual(y2-y1,r['size'])
            self.assertNotEqual(r['review_status'],'pending_visual_crop_review')

    def test_similarity_bands_partition_full_cosine(self):
        features=rows(OUT/'features.csv')
        bands=['low_0_025','mid_025_050','mid_050_090','high_090_177']
        checked=0
        for r in features:
            for ref in ['global','surface']:
                if not r['similarity_'+ref]:continue
                total=sum(float(r['band_contribution_'+ref+'_'+b]) for b in bands)
                self.assertAlmostEqual(total,float(r['similarity_'+ref]),places=9);checked+=1
        self.assertGreater(checked,0)

    def test_light_off_surface_tag_gaps_are_positive(self):
        gaps=rows(OUT/'04_similarity/within_image_gaps.csv')
        selected=[float(r['gap_surface_minus_other']) for r in gaps if r['lighting']=='light_off'
                  and r['reference']=='global' and r['other_category'] in ['tag_plain','tag_text']]
        self.assertEqual(len(selected),8)
        self.assertTrue(all(v>0 for v in selected))

    def test_reproduction_and_annotation_totals(self):
        checks=rows(OUT/'06_mask_analysis/reproduction_checks.csv')
        self.assertEqual(len(checks),21)
        self.assertTrue(all(r['all_four_masks_pixel_identical']=='True' for r in checks))
        retention=rows(OUT/'06_mask_analysis/all_annotation_retention.csv')
        self.assertEqual(len(retention),54*4)
        legacy=[r for r in retention if r['method']=='legacy256']
        self.assertEqual(sum(r['center_kept']=='True' for r in legacy),50)
        self.assertEqual(sum(float(r['retained_fraction'])>=.95 for r in legacy),49)


if __name__=='__main__':unittest.main()
