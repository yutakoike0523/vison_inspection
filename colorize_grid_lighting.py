"""Overlay two adaptive histogram brightness bands on grid-illuminated images.

The two bands are selected independently per image so exposure differences do
not make image15's absolute brightness thresholds dominate every other image.
This is a visualization of brightness structure, not defect classification.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
INPUT = ROOT/'data20260903/data20260903'
OUT = ROOT/'outputs/all_grid_images_histogram_color'
BLUE = np.array([13, 82, 194], dtype=float)/255
ORANGE = np.array([240, 77, 13], dtype=float)/255


def peak_in_range(smooth, lower, upper):
    lower=max(0, int(np.floor(lower)))
    upper=min(255, int(np.ceil(upper)))
    if lower>=upper:
        return None
    return int(lower+np.argmax(smooth[lower:upper+1]))


def select_bands(image):
    """Return two brightness-centered bands using the lower/upper half of data."""
    counts=np.bincount(image.ravel(), minlength=256)
    smooth=gaussian_filter1d(counts.astype(float), 2)
    # Avoid extreme black/white areas such as the image edge or saturated paper.
    # The central part of the image distribution is split into its darker and
    # brighter half, leaving a small gap around the median.
    q10,q45,q55,q80=np.percentile(image,[10,45,55,80])
    dark=peak_in_range(smooth,q10,q45)
    bright=peak_in_range(smooth,q55,q80)
    # When a peak lands too close to the central gap, retain separation.
    if bright-dark<12:
        dark=peak_in_range(smooth,q10,q45-4)
        bright=peak_in_range(smooth,q55+4,q80)
    if dark is None or bright is None or bright<=dark:
        raise ValueError('Two separated brightness bands could not be selected.')
    # The decision boundary for a complete two-class separation is the valley
    # (minimum smoothed histogram count) between the two peak positions.
    boundary=int(dark+np.argmin(smooth[dark:bright+1]))
    # Keep the two bands disjoint even when the peaks are close together.
    half_width=min(int(np.clip(round((bright-dark)/4),4,12)),(bright-dark-1)//2)
    if half_width<1:
        raise ValueError('The two brightness peaks are too close to form separate bands.')
    return dict(dark_peak=dark, bright_peak=bright, decision_boundary=boundary, half_width=half_width,
                dark_range=[max(0,dark-half_width), min(255,dark+half_width)],
                bright_range=[max(0,bright-half_width), min(255,bright+half_width)],
                quantiles=[float(q10),float(q45),float(q55),float(q80)], counts=counts)


def overlay(image, bands, alpha=.68):
    rgb=np.dstack([image,image,image]).astype(float)/255
    d0,d1=bands['dark_range'];b0,b1=bands['bright_range']
    masks=[(image>=d0)&(image<=d1),(image>=b0)&(image<=b1)]
    assert not (masks[0]&masks[1]).any()
    for mask,color in zip(masks,[BLUE,ORANGE]):
        rgb[mask]=(1-alpha)*rgb[mask]+alpha*color
    return (rgb*255).round().astype(np.uint8), masks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_files=[INPUT/f'image{i}.png' for i in range(2,19)]
    records=[];thumbs=[]
    for source in source_files:
        image=np.asarray(Image.open(source).convert('L'),dtype=np.uint8)
        bands=select_bands(image)
        colored,masks=overlay(image,bands)
        Image.fromarray(colored).save(OUT/f'{source.stem}_two_brightness_bands.png')
        record={'image':source.name,**{k:v for k,v in bands.items() if k!='counts'},
                'dark_pixels':int(masks[0].sum()),'bright_pixels':int(masks[1].sum()),
                'method':'histogram smoothed with sigma=2; central 10-80% brightness span split at 45-55%; band half-width=min(12,max(4,round((bright-dark)/4)))'}
        records.append(record)
        thumb=Image.fromarray(colored).resize((384,288),Image.Resampling.LANCZOS).convert('RGB')
        canvas=Image.new('RGB',(384,316),'white');canvas.paste(thumb,(0,28))
        draw=ImageDraw.Draw(canvas)
        draw.text((8,7),f"{source.stem}: blue {record['dark_range'][0]}-{record['dark_range'][1]}, orange {record['bright_range'][0]}-{record['bright_range'][1]}",fill='black')
        thumbs.append(canvas)
    sheet=Image.new('RGB',(384*3,316*6),'white')
    for i,thumb in enumerate(thumbs):sheet.paste(thumb,((i%3)*384,(i//3)*316))
    sheet.save(OUT/'contact_sheet.png')
    (OUT/'band_settings.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savetxt(OUT/'band_settings.csv',np.array([[int(r['image'][5:-4]),r['dark_peak'],r['bright_peak'],r['decision_boundary'],*r['dark_range'],*r['bright_range'],r['dark_pixels'],r['bright_pixels']] for r in records]),delimiter=',',fmt='%d',header='image_number,dark_peak,bright_peak,decision_boundary,dark_min,dark_max,bright_min,bright_max,dark_pixels,bright_pixels',comments='')
    print(json.dumps(records,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
