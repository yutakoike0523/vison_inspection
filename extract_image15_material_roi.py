"""Transfer the user's red ROI annotation to the original image coordinates.

This is deterministic scientific ROI extraction, not image synthesis. The source
pixels are preserved exactly; the annotation defines a separate binary mask.
The known blue rectangle calibrates the annotated Matplotlib figure coordinates.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label, map_coordinates

BASE=Path(__file__).resolve().parent
SOURCE=BASE/'data20260903/data20260903/image15.png'
REFERENCE=BASE/'outputs/grid_detection_image15/01_detection_roi.png'
ANNOTATION=Path('C:/Users/kyuta/Downloads/01_detection_roi.png')
OUT=BASE/'outputs/image15_material_roi'


def line_centers(counts):
    indices=np.flatnonzero(counts>counts.max()*.8)
    groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    if len(groups)!=2:
        raise RuntimeError('Expected two sides of the blue calibration rectangle.')
    return [float(g.mean()) for g in groups]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=np.asarray(Image.open(SOURCE).convert('RGB'))
    reference=np.asarray(Image.open(REFERENCE).convert('RGB')).astype(int)
    annotated=np.asarray(Image.open(ANNOTATION).convert('RGB')).astype(int)
    if reference.shape!=annotated.shape:
        raise ValueError('Annotated and reference figures must have matching dimensions.')
    r,g,b=reference.transpose(2,0,1)
    blue=(b-r>70)&(g-r>35)&(b-g>20)
    left,right=line_centers(blue.sum(axis=0))
    top,bottom=line_centers(blue.sum(axis=1))
    # Original rectangle: (99.5,119.5), width=1000, height=1280.
    sx=(right-left)/1000;sy=(bottom-top)/1280
    tx=left-sx*99.5;ty=top-sy*119.5
    r,g,b=annotated.transpose(2,0,1)
    red=(r>160)&(r-g>80)&(r-b>60)
    interior=binary_fill_holes(red)&~red
    labels,count=label(interior)
    areas=np.bincount(labels.ravel());areas[0]=0
    if count==0 or areas.max()<10000:
        raise RuntimeError('No sufficiently large closed red region was found.')
    interior=labels==areas.argmax()
    h,w=source.shape[:2];yy,xx=np.indices((h,w))
    mask=map_coordinates(interior.astype(np.uint8),[sy*yy+ty,sx*xx+tx],order=0,mode='constant',cval=0).astype(bool)
    ys,xs=np.nonzero(mask)
    box=(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1))
    x1,y1,x2,y2=box
    # White RGB outside also makes viewers that ignore alpha show the cutout.
    rgba=np.dstack([np.where(mask[:,:,None],source,255).astype(np.uint8),mask.astype(np.uint8)*255])
    Image.fromarray(mask.astype(np.uint8)*255).save(OUT/'image15_material_mask.png')
    Image.fromarray(rgba).save(OUT/'image15_material_cutout_fullsize.png')
    Image.fromarray(rgba[y1:y2,x1:x2]).save(OUT/'image15_material_cutout.png')

    # Figures use original source pixels, so annotation, axes and the old blue
    # rectangle are absent from every exported crop.
    plt.rcParams.update({'font.family':'Yu Gothic','font.size':12,'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
    axes[0].imshow(source)
    axes[0].contour(mask.astype(float),levels=[.5],colors=['#e41a32'],linewidths=1.5)
    axes[0].set_title('赤線から作成した対象範囲')
    axes[1].imshow(mask,cmap='gray',vmin=0,vmax=1)
    axes[1].set_title('解析用マスク：白＝対象、黒＝対象外')
    axes[2].imshow(rgba)
    axes[2].set_title('元画像の画素を保持して切り出し')
    for ax in axes:ax.set_axis_off()
    fig.suptitle('image15：長方形ではなく、指定した輪郭に沿う対象範囲')
    fig.savefig(OUT/'material_roi_steps.png',dpi=160,bbox_inches='tight')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    ax.imshow(source)
    ax.contour(mask.astype(float),levels=[.5],colors=['#e41a32'],linewidths=1.5)
    ax.set(title='image15：指定した赤線に対応する解析対象範囲',xlabel='元画像 x [pixel]',ylabel='元画像 y [pixel]')
    fig.savefig(OUT/'image15_material_roi_overlay.png',dpi=155,bbox_inches='tight')
    plt.close(fig)

    # Verify all retained source values and transparent excluded pixels.
    saved=np.asarray(Image.open(OUT/'image15_material_cutout_fullsize.png'))
    assert np.array_equal(saved[:,:,:3][mask],source[mask])
    assert np.all(saved[:,:,:3][~mask]==255)
    assert np.array_equal(saved[:,:,3]>0,mask)
    cropped=np.asarray(Image.open(OUT/'image15_material_cutout.png'))
    assert np.array_equal(cropped,saved[y1:y2,x1:x2])
    assert mask[800,600] and not mask[600,1700] and not mask[1300,1900]
    meta={'source':str(SOURCE),'annotation':str(ANNOTATION),'reference':str(REFERENCE),
          'method':'red enclosed interior, excluding stroke; affine calibration from existing blue rectangle',
          'source_size_wh':[w,h],'figure_mapping':{'sx':sx,'sy':sy,'tx':tx,'ty':ty},
          'bounding_box_xyxy_exclusive':box,'crop_origin_xy':[x1,y1],
          'included_pixels':int(mask.sum()),'included_fraction':float(mask.mean()),
          'original_pixels_preserved':True,
          'notes':['User-drawn ROI, not an automatically segmented material boundary.',
                   'Thick red stroke is excluded, so the region follows its inner edge.',
                   'Use original image and binary mask for analysis; do not treat transparent outside pixels as measured zeros.',
                   'Accept a local analysis crop only when its entire footprint is inside the mask.']}
    (OUT/'roi_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'crop_box':box,'included_fraction':mask.mean(),'retained_pixels':int(mask.sum()),'pixel_verification':'passed'},ensure_ascii=False))


if __name__=='__main__':main()
