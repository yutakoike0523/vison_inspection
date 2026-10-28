"""Analyze existing FFT decisions; no modifications to prior scripts or artifacts."""
from pathlib import Path
import argparse
import csv
import json
import hashlib
import math
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fft2_cosine_similarity as fft
import point_anomaly_fft_mask as fm
import point_anomaly_contrast as pc

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/fft_region_analysis'
SOURCE=ROOT/'data20260903/data20260903'
PREVIOUS=ROOT/'outputs/point_anomaly_fft_mask'
NO_LIGHT={1,19,20,21}
FOLDERS=['01_roi','02_spectra','03_frequency_profiles','04_similarity','05_lighting_comparison','06_mask_analysis','07_failure_cases']


def setup():
    for folder in FOLDERS:
        (OUT/folder).mkdir(parents=True,exist_ok=True)


def overview():
    setup(); panels=[]
    for i in range(1,22):
        path=SOURCE/f'image{i}.png'
        if not path.exists(): continue
        rgb=pc.read_image(path);im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
        step=64 if i==18 else 256
        for x in range(0,im.width,step):
            draw.line((x,0,x,im.height),fill='yellow',width=1)
            draw.text((x+2,2),str(x),fill='red',stroke_width=1,stroke_fill='white')
        for y in range(0,im.height,step):
            draw.line((0,y,im.width,y),fill='yellow',width=1)
            draw.text((2,y+2),str(y),fill='red',stroke_width=1,stroke_fill='white')
        im=im.resize((512,384))
        panel=Image.new('RGB',(512,410),'#222222');panel.paste(im,(0,26))
        ImageDraw.Draw(panel).text((8,7),f'image{i} | '+('LIGHT OFF' if i in NO_LIGHT else 'LIGHT ON')+f' | {rgb.shape[1]}x{rgb.shape[0]}',fill='white')
        panel.save(OUT/'01_roi'/f'image{i}_coordinate_preview.png');panels.append(panel)
    for start in range(0,len(panels),6):
        sheet=Image.new('RGB',(1536,820),'#222222')
        for j,panel in enumerate(panels[start:start+6]): sheet.paste(panel,((j%3)*512,(j//3)*410))
        sheet.save(OUT/'01_roi'/f'overview_{start//6+1}.png')


SURFACES={
1:[(1100,100),(1200,600),(500,1150)],2:[(950,150),(1050,1050),(1450,700)],
3:[(1000,150),(1350,500),(700,1100)],4:[(1000,150),(1400,600),(600,1000)],
5:[(1200,100),(700,800),(1500,1000)],6:[(1100,100),(700,900),(1500,1100)],
7:[(1000,100),(1500,500),(1000,900)],8:[(800,100),(900,500),(700,950)],
9:[(1200,100),(1500,900),(650,1000)],10:[(1400,100),(600,900),(1500,900)],
11:[(900,180),(1000,600),(900,1000)],12:[(1100,100),(1100,600),(1100,1050)],
13:[(900,40),(450,700),(1350,700)],14:[(1000,100),(500,600),(950,850)],
15:[(350,400),(1100,400),(850,1100)],16:[(800,100),(400,500),(900,1100)],
17:[(300,250),(200,1000),(1050,1150)],19:[(150,500),(600,750),(1000,1150)],
20:[(800,80),(300,400),(900,600)],21:[(400,400),(700,30),(800,1100)]}
TAGS={1:[('tag_plain',20,250,128),('tag_edge_mixed',80,160,256)],
2:[('tag_text',150,100,256)],3:[('tag_text',300,80,256)],4:[('tag_text',350,120,256)],
5:[('tag_text',300,60,256),('tag_plain',750,280,128)],6:[('tag_text',350,120,256)],
7:[('tag_text',650,450,256)],8:[('tag_text',300,350,256)],9:[('tag_text',450,100,256)],
10:[('tag_text',500,70,256)],11:[('tag_text',250,650,256)],12:[('tag_text',300,400,256)],
13:[('tag_text',100,0,256),('tag_plain',450,50,128)],14:[('tag_text',200,20,256)],
15:[('tag_text',1700,400,256),('tag_plain',1800,250,128)],
16:[('tag_plain',1780,600,256),('tag_edge_mixed',1700,400,256)],
17:[('tag_text',1600,650,256),('tag_plain',1550,300,128)],18:[('tag_edge_mixed',300,100,128)],
19:[('tag_text',1650,800,256)],20:[('tag_plain',1780,800,256),('tag_edge_mixed',1600,600,256)],
21:[('tag_text',100,0,256),('tag_plain',450,50,128)]}
OUTSIDE={3:[(0,1408,128)],4:[(0,1408,128)],7:[(0,300,128),(512,1280,256)],
8:[(1900,500,128),(600,1280,256)],9:[(0,1408,128)],11:[(0,0,128)],12:[(0,1000,128)],
15:[(0,0,128)],17:[(0,0,128)],19:[(0,0,256)]}


def annotations():
    data=json.loads((SOURCE/'全傷.json').read_text(encoding='utf-8-sig'))
    result={}
    for item in data['images']:
        shapes=[]
        for j,s in enumerate(item['shapes']):
            p=np.array(s['points'])
            shapes.append({'id':j+1,'label':s['label'],'box':[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]})
        result[Path(item['imagePath']).stem]=shapes
    return result


def generate_rois():
    setup();anns=annotations();regions=[];unavailable=[]
    for i in range(1,22):
        name=f'image{i}';path=SOURCE/(name+'.png')
        if not path.is_file():
            unavailable.append({'image':name,'reason':'missing image'});continue
        rgb=pc.read_image(path);h,w=rgb.shape[:2]
        specs=[('surface',x,y,256) for x,y in SURFACES.get(i,[])]+TAGS.get(i,[])+[('outside',x,y,n) for x,y,n in OUTSIDE.get(i,[])]
        if i==18:specs+=[('surface',100,0,128),('surface',240,40,128)]
        for k,(category,x,y,n) in enumerate(specs):
            for size in ([128,256] if n==256 else [128]):
                xx,yy=x+(n-size)//2,y+(n-size)//2
                box=[xx,yy,xx+size,yy+size]
                overlaps=[a['id'] for a in anns[name] if fft.intersect(box,a['box'])>0]
                if category=='surface' and overlaps:
                    unavailable.append({'image':name,'category':category,'box':box,'reason':'intersects existing defect annotation','annotation_ids':overlaps});continue
                assert 0<=xx<xx+size<=w and 0<=yy<yy+size<=h
                regions.append({'id':f'{name}_r{k+1}_{size}','image':name,'category':category,'box':box,'size':size,
                    'selection':'manual_coordinates_from_visual_inspection','review_status':'pending_visual_crop_review',
                    'reviewer':'assistant','human_confirmed':False,'annotation_overlaps':overlaps,'annotation_id':None,
                    'note':'surface means visually plausible unannotated surface, not certified defect-free' if category=='surface' else 'mixed' if category=='tag_edge_mixed' else 'provisional visual category'})
        for a in anns[name]:
            for size in [128,256]:
                if size>min(h,w):continue
                cx,cy=(a['box'][0]+a['box'][2])/2,(a['box'][1]+a['box'][3])/2
                xx=min(max(0,int(cx-size/2)),w-size);yy=min(max(0,int(cy-size/2)),h-size)
                regions.append({'id':f"{name}_damage{a['id']}_{size}",'image':name,'category':'damage','box':[xx,yy,xx+size,yy+size],'size':size,
                    'selection':'annotation_centered','review_status':'annotation_derived','reviewer':'user_annotation',
                    'human_confirmed':False,'annotation_overlaps':[b['id'] for b in anns[name] if fft.intersect([xx,yy,xx+size,yy+size],b['box'])>0],
                    'annotation_id':a['id'],'damage_label':a['label'],'note':'Context block can include background; rectangle is not a defect contour'})
        for cat in ['surface','tag_plain','tag_text','outside']:
            for size in [128,256]:
                if not any(r['image']==name and r['category']==cat and r['size']==size for r in regions):
                    unavailable.append({'image':name,'category':cat,'size':size,'reason':'no confidently selected pure region at this size'})
    pc.write_json(OUT/'roi_definitions.json',{'version':1,'lighting_off':[f'image{i}' for i in sorted(NO_LIGHT)],
        'label_policy':'Assistant visually selected provisional ROI labels; not human-validated ground truth. Edit coordinates/category/review_status before --run to revise.',
        'regions':regions,'unavailable':unavailable})
    roi_previews(regions)


def roi_previews(regions):
    for i in range(1,22):
        name=f'image{i}';rgb=pc.read_image(SOURCE/(name+'.png'))
        chosen=[r for r in regions if r['image']==name and r['category']!='damage']
        canvas=Image.fromarray(rgb);draw=ImageDraw.Draw(canvas)
        panels=[]
        for r in chosen:
            x1,y1,x2,y2=r['box'];draw.rectangle(r['box'],outline='red',width=3);draw.text((x1,y1),r['id'],fill='red')
            crop=Image.fromarray(rgb[y1:y2,x1:x2]).resize((192,192),Image.Resampling.NEAREST)
            p=Image.new('RGB',(240,228),'#222222');p.paste(crop,(24,36));d=ImageDraw.Draw(p)
            d.text((5,3),r['id'],fill='white');d.text((5,18),r['category'],fill='white');panels.append(p)
        sheet=Image.new('RGB',(240*5,228*math.ceil(len(panels)/5)),'#222222')
        for j,p in enumerate(panels):sheet.paste(p,((j%5)*240,(j//5)*228))
        sheet.save(OUT/'01_roi'/f'{name}_roi_review.png')
        canvas.resize((1024,round(rgb.shape[0]*1024/rgb.shape[1]))).save(OUT/'01_roi'/f'{name}_roi_locations.png')


def apply_visual_review():
    data=json.loads((OUT/'roi_definitions.json').read_text(encoding='utf-8'))
    changes={}
    for i in [2,3,4,5,6,21]:changes[f'image{i}_r1_256']='tag_edge_mixed'
    for i in [3,4,7,8,11]:changes[f'image{i}_r4_256']='tag_edge_mixed'
    changes.update({'image3_r3_256':'outside_edge_mixed','image12_r5_128':'unclassified',
        'image7_r2_128':'uncertain_surface','image7_r2_256':'uncertain_surface',
        'image15_r2_128':'tag_edge_mixed','image15_r2_256':'tag_edge_mixed',
        'image15_r5_128':'tag_text','image15_r6_128':'outside_edge_mixed',
        'image16_r3_256':'uncertain_surface','image17_r5_128':'tag_edge_mixed',
        'image18_r3_128':'uncertain_surface','image19_r3_256':'tag_edge_mixed',
        'image21_r2_128':'tag_edge_mixed','image21_r2_256':'tag_edge_mixed','image1_r5_128':'tag_plain'})
    for r in data['regions']:
        if r['category']=='damage':continue
        if r['id'] in changes:
            r['original_proposed_category']=r['category'];r['category']=changes[r['id']]
            r['note']='Crop reviewed: mixed boundary or uncertain content; not a pure surface reference' if 'mixed' in r['category'] or 'uncertain' in r['category'] else 'Category corrected after viewing actual crop'
        r['review_status']='assistant_visual_reviewed'
    for i,x,y,serial in [(3,1450,900,1),(21,600,500,1),(21,900,600,2)]:
        for n in [128,256]:
            xx,yy=x+(256-n)//2,y+(256-n)//2
            data['regions'].append({'id':f'image{i}_extra{serial}_{n}','image':f'image{i}','category':'surface',
                'box':[xx,yy,xx+n,yy+n],'size':n,'selection':'manual_replacement','review_status':'pending_visual_crop_review',
                'reviewer':'assistant','human_confirmed':False,'annotation_overlaps':[],'annotation_id':None,'note':'Additional visually selected surface, pending crop inspection'})
    pc.write_json(OUT/'roi_definitions.json',data);roi_previews(data['regions'])


COMMON_AXIS=np.linspace(-.125,.125,129)
FY,FX=np.meshgrid(COMMON_AXIS,COMMON_AXIS,indexing='ij')
RAD=np.hypot(FX,FY)
CMASK=fft.comparison_mask(129,5)
BANDS={'low_0_025':CMASK&(RAD<.025),'mid_025_050':CMASK&(RAD>=.025)&(RAD<.05),
       'mid_050_090':CMASK&(RAD>=.05)&(RAD<.09),'high_090_177':CMASK&(RAD>=.09)}
CATEGORIES=['surface','tag_plain','tag_text','outside','damage','tag_edge_mixed','outside_edge_mixed','uncertain_surface','unclassified']
COLORS={'surface':'#247a36','tag_plain':'#175cc1','tag_text':'#8a38b8','outside':'#df731e','damage':'#d32828',
        'tag_edge_mixed':'#808080','outside_edge_mixed':'#999966','uncertain_surface':'#999999','unclassified':'#444444'}


def group_for(name):
    number=int(name[5:])
    return 'light_off' if number in NO_LIGHT else 'light_on_lowres' if number==18 else 'light_on_highres'


def native_profile(amplitude):
    n=amplitude.shape[0];f=np.fft.fftshift(np.fft.fftfreq(n));fy,fx=np.meshgrid(f,f,indexing='ij');radius=np.hypot(fx,fy)
    edges=np.linspace(0,np.sqrt(.5),65);idx=np.minimum(np.digitize(radius.ravel(),edges)-1,63)
    power=amplitude.ravel()**2;counts=np.bincount(idx,minlength=64)
    means=np.divide(np.bincount(idx,weights=power,minlength=64),counts,out=np.zeros(64),where=counts>0)
    return (edges[:-1]+edges[1:])/2,means


def stats(values):
    v=np.array([x for x in values if x is not None and np.isfinite(x)],float)
    return {'count':len(v),'mean':float(v.mean()) if len(v) else None,'median':float(np.median(v)) if len(v) else None,
            'std':float(v.std()) if len(v) else None,'min':float(v.min()) if len(v) else None,'max':float(v.max()) if len(v) else None}


def compute_roi_features(regions):
    rows=[];cache={};globals_by_image={};profiles=[];peak_rows=[]
    for i in range(1,22):
        name=f'image{i}';path=SOURCE/(name+'.png')
        if not path.exists():continue
        gray=fft.read_gray(path);_,ga=fft.fft_amplitude(gray)
        global_common=fft.common_spectrum(ga,129,.125);globals_by_image[name]=global_common
        chosen=[r for r in regions if r['image']==name]
        for r in chosen:
            x1,y1,x2,y2=r['box'];patch=gray[y1:y2,x1:x2]
            assert patch.shape==(r['size'],r['size'])
            raw,a=fft.fft_amplitude(patch);common=fft.common_spectrum(a,129,.125)
            f,profile=native_profile(a)
            peaks=fft.peak_candidates(raw,3,99,7)[:20]
            for j,(py,px) in enumerate(peaks):
                peak_rows.append({'roi':r['id'],'rank':j+1,'fx':(px-r['size']//2)/r['size'],'fy':(py-r['size']//2)/r['size'],
                                  'normalized_amplitude':float(a[py,px]),'raw_amplitude':float(raw[py,px])})
            for freq,power in zip(f,profile):profiles.append({'roi':r['id'],'frequency':float(freq),'mean_power':float(power)})
            cache[r['id']]={'patch':patch,'raw':raw,'amplitude':a,'common':common,'profile':profile,'frequencies':f,'peaks':peaks}
            pc.save_image(OUT/'01_roi'/(r['id']+'_native.png'),patch)
            np.savez_compressed(OUT/'02_spectra'/(r['id']+'.npz'),amplitude=a,common=common)
        for r in chosen:
            a=cache[r['id']];v=fft.vector(a['common'],CMASK,'log')
            normals=[other for other in chosen if other['category']=='surface' and other['size']==r['size'] and other['review_status'] in ['assistant_visual_reviewed','human_confirmed']
                     and other['id']!=r['id'] and fft.intersect(other['box'],r['box'])==0]
            ref=np.median(np.stack([cache[n['id']]['common'] for n in normals]),axis=0) if normals else None
            refs={'global':global_common,'surface':ref}
            row={'roi':r['id'],'image':name,'lighting':group_for(name),'size':r['size'],'category':r['category'],
                'review_status':r['review_status'],'human_confirmed':r['human_confirmed'],'annotation_id':r['annotation_id'],
                'damage_label':r.get('damage_label',''),'x1':r['box'][0],'y1':r['box'][1],'x2':r['box'][2],'y2':r['box'][3],
                'surface_reference_count':len(normals),'surface_reference_ids':'|'.join(n['id'] for n in normals),
                'mean_brightness':float(a['patch'].mean()),'brightness_std':float(a['patch'].std()),
                'saturated_fraction':float(np.mean(a['patch']==255)),'peaks_count':len(a['peaks'])}
            row['peak_energy_fraction']=float(sum(a['amplitude'][y,x]**2 for y,x in a['peaks'])/max(float(np.sum(a['amplitude']**2)),1e-30))
            for refname,reference in refs.items():
                if reference is None:
                    row['similarity_'+refname]=None;row['linear_similarity_'+refname]=None
                    for band in BANDS:
                        for prefix in ['band_similarity_','band_contribution_','without_band_']:row[prefix+refname+'_'+band]=None
                    continue
                rv=fft.vector(reference,CMASK,'log');row['similarity_'+refname]=fft.cosine(v,rv)
                row['linear_similarity_'+refname]=fft.cosine(fft.vector(a['common'],CMASK,'linear'),fft.vector(reference,CMASK,'linear'))
                denom=float(np.linalg.norm(v)*np.linalg.norm(rv))
                for band,mask in BANDS.items():
                    av=fft.vector(a['common'],mask,'log');bv=fft.vector(reference,mask,'log')
                    row['band_similarity_'+refname+'_'+band]=fft.cosine(av,bv)
                    row['band_contribution_'+refname+'_'+band]=float(np.dot(av,bv)/denom) if denom>1e-15 else None
                    row['without_band_'+refname+'_'+band]=fft.cosine(fft.vector(a['common'],CMASK&~mask,'log'),fft.vector(reference,CMASK&~mask,'log'))
                if denom>1e-15:
                    assert abs(sum(row['band_contribution_'+refname+'_'+b] for b in BANDS)-row['similarity_'+refname])<1e-10
            rows.append(row)
        print('ROI spectra:',name,flush=True)
    pc.csv_out(OUT/'features.csv',rows);pc.csv_out(OUT/'03_frequency_profiles/radial_profiles.csv',profiles);pc.csv_out(OUT/'02_spectra/peaks.csv',peak_rows)
    return rows,cache


def spectrum_figures(regions,rows,cache):
    vmax=float(np.quantile(np.concatenate([np.log1p(a['amplitude']).ravel() for a in cache.values()]),.999))
    pc.write_json(OUT/'02_spectra/display_settings.json',{'transform':'log1p(amplitude/window_sum)','vmin':0,'vmax':vmax,
                   'shared_across_all_images_and_regions':True,'center_range_cycles_per_pixel':[-.125,.125]})
    for i in range(1,22):
        name=f'image{i}'
        for size in [128,256]:
            chosen=[]
            for category in CATEGORIES:
                found=[r for r in regions if r['image']==name and r['size']==size and r['category']==category]
                if found:chosen.append(found[0])
            if not chosen:continue
            # A shared intensity axis makes the raw grayscale distributions
            # directly comparable across every ROI shown in the figure.
            histogram_bins=np.linspace(0,256,65)
            histogram_density=[]
            for r in chosen:
                counts,_=np.histogram(cache[r['id']]['patch'],bins=histogram_bins,density=True)
                histogram_density.append(counts)
            histogram_ymax=max(max(counts) for counts in histogram_density)*1.08
            fig,axes=plt.subplots(4,len(chosen),figsize=(3.2*len(chosen),11.5),squeeze=False,layout='constrained')
            for col,r in enumerate(chosen):
                a=cache[r['id']];axes[0,col].imshow(a['patch'],cmap='gray',vmin=0,vmax=255)
                axes[0,col].set_title(r['category']+'\n'+r['id'],fontsize=9)
                axes[1,col].imshow(np.log1p(a['amplitude']),cmap='magma',vmin=0,vmax=vmax,extent=[-.5,.5,.5,-.5])
                im=axes[2,col].imshow(np.log1p(a['amplitude']),cmap='magma',vmin=0,vmax=vmax,extent=[-.5,.5,.5,-.5])
                axes[2,col].set(xlim=(-.125,.125),ylim=(.125,-.125),xlabel='fx [cycles/pixel]')
                points=[((x-size//2)/size,(y-size//2)/size) for y,x in a['peaks']]
                if points:axes[2,col].scatter(*zip(*points),facecolors='none',edgecolors='cyan',s=25,linewidths=.6)
                centers=(histogram_bins[:-1]+histogram_bins[1:])/2
                axes[3,col].step(centers,histogram_density[col],where='mid',color='#303030',lw=1.2)
                axes[3,col].fill_between(centers,histogram_density[col],step='mid',color='#888888',alpha=.35)
                axes[3,col].axvline(float(a['patch'].mean()),color='red',ls='--',lw=.9,label=f"mean={a['patch'].mean():.1f}")
                axes[3,col].set(xlim=(0,255),ylim=(0,histogram_ymax),xlabel='Grayscale intensity',ylabel='Density')
                axes[3,col].legend(fontsize=7,loc='upper right')
                axes[0,col].axis('off');axes[1,col].set_xlabel('fx [cycles/pixel]')
            axes[1,0].set_ylabel('Full spectrum fy');axes[2,0].set_ylabel('Center spectrum fy')
            fig.colorbar(im,ax=axes[1:3].ravel().tolist(),shrink=.6,label='log1p normalized amplitude (shared scale)')
            fig.suptitle(f'{name} | {group_for(name)} | {size}px | ROI categories are provisional',fontsize=13)
            fig.savefig(OUT/'02_spectra'/f'{name}_{size}_comparison.png',dpi=110);plt.close(fig)
            fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
            for r in chosen:
                a=cache[r['id']];ax.semilogy(a['frequencies'],np.maximum(a['profile'],1e-12),label=r['category'],color=COLORS[r['category']])
            ax.set(xlim=(0,.5),xlabel='Radial frequency [cycles/pixel]',ylabel='Mean squared normalized amplitude',title=f'{name} | {size}px');ax.legend(fontsize=8);ax.grid(alpha=.25)
            fig.savefig(OUT/'03_frequency_profiles'/f'{name}_{size}_profiles.png',dpi=140);plt.close(fig)


def summarize_similarity(rows):
    summary=[];lighting=[];gaps=[];bands=[]
    for image in sorted({r['image'] for r in rows},key=lambda n:int(n[5:])):
        for size in [128,256]:
            for ref in ['global','surface']:
                for category in CATEGORIES:
                    part=[r for r in rows if r['image']==image and r['size']==size and r['category']==category]
                    if part:summary.append({'image':image,'lighting':group_for(image),'size':size,'reference':ref,'category':category,**stats([r['similarity_'+ref] for r in part])})
                sur=[r for r in rows if r['image']==image and r['size']==size and r['category']=='surface']
                for cat in ['tag_plain','tag_text','outside','damage','tag_edge_mixed']:
                    other=[r for r in rows if r['image']==image and r['size']==size and r['category']==cat]
                    sa=stats([r['similarity_'+ref] for r in sur]);sb=stats([r['similarity_'+ref] for r in other])
                    if not sa['count'] or not sb['count']:continue
                    gap=sa['mean']-sb['mean']
                    gaps.append({'image':image,'lighting':group_for(image),'size':size,'reference':ref,'other_category':cat,
                                 'surface_count':sa['count'],'other_count':sb['count'],'surface_mean':sa['mean'],'other_mean':sb['mean'],'gap_surface_minus_other':gap})
                    for band in BANDS:
                        aa=stats([r['band_similarity_'+ref+'_'+band] for r in sur]);bb=stats([r['band_similarity_'+ref+'_'+band] for r in other])
                        ca=stats([r['band_contribution_'+ref+'_'+band] for r in sur]);cb=stats([r['band_contribution_'+ref+'_'+band] for r in other])
                        wa=stats([r['without_band_'+ref+'_'+band] for r in sur]);wb=stats([r['without_band_'+ref+'_'+band] for r in other])
                        if aa['count'] and bb['count'] and wa['count'] and wb['count']:
                            bands.append({'image':image,'lighting':group_for(image),'size':size,'reference':ref,'other_category':cat,'band':band,
                                          'full_gap':gap,'band_similarity_gap':aa['mean']-bb['mean'],
                                          'additive_contribution_to_gap':ca['mean']-cb['mean'],
                                          'gap_without_band':wa['mean']-wb['mean'],'gap_loss_when_removed':gap-(wa['mean']-wb['mean'])})
    for group in ['light_off','light_on_highres','light_on_lowres']:
        for size in [128,256]:
            for ref in ['global','surface']:
                for cat in CATEGORIES:
                    part=[r for r in summary if r['lighting']==group and r['size']==size and r['reference']==ref and r['category']==cat and r['mean'] is not None]
                    if part:lighting.append({'lighting':group,'size':size,'reference':ref,'category':cat,'roi_count':sum(r['count'] for r in part),
                                             'aggregation':'equal-weight image means',**stats([r['mean'] for r in part])})
    pc.csv_out(OUT/'similarity_summary.csv',summary);pc.csv_out(OUT/'lighting_summary.csv',lighting)
    pc.csv_out(OUT/'04_similarity/within_image_gaps.csv',gaps);pc.csv_out(OUT/'03_frequency_profiles/band_attribution.csv',bands)
    for size in [128,256]:
        fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained',sharey=True)
        for row,group in enumerate(['light_off','light_on_highres','light_on_lowres']):
            for col,ref in enumerate(['global','surface']):
                ax=axes[row,col];cats=['surface','tag_plain','tag_text','outside','damage','tag_edge_mixed']
                values=[[r['similarity_'+ref] for r in rows if r['lighting']==group and r['size']==size and r['category']==c and r['similarity_'+ref] is not None] for c in cats]
                for j,vals in enumerate(values):
                    if vals:
                        ax.boxplot([vals],positions=[j],widths=.5,showfliers=False)
                        ax.scatter(np.full(len(vals),j)+np.linspace(-.15,.15,len(vals)),vals,s=10,alpha=.65,color=COLORS[cats[j]])
                ax.set(xticks=range(len(cats)),xticklabels=cats,ylim=(0,1.03),title=group+' | ref='+ref,ylabel='Cosine similarity')
                ax.tick_params(axis='x',rotation=35);ax.axhline(.9,color='gray',ls='--',lw=.7);ax.grid(axis='y',alpha=.2)
        fig.suptitle(f'{size}px | Provisional ROI labels; overlapping/context ROIs not independent')
        fig.savefig(OUT/'04_similarity'/f'boxplots_{size}.png',dpi=130);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained',sharex=True)
        for ax,group in zip(axes,['light_off','light_on_highres','light_on_lowres']):
            for cat in ['surface','tag_plain','tag_text','outside']:
                values=[r['similarity_global'] for r in rows if r['size']==size and r['lighting']==group and r['category']==cat and r['similarity_global'] is not None]
                if values:ax.hist(values,bins=np.linspace(0,1,26),histtype='step',label=f'{cat} n={len(values)}',color=COLORS[cat])
            ax.set(title=group,xlabel='Global-reference cosine',ylabel='ROI count');ax.legend(fontsize=7)
        fig.savefig(OUT/'04_similarity'/f'histograms_{size}.png',dpi=130);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,ref in zip(axes,['global','surface']):
            for category,marker in [('tag_plain','o'),('tag_text','s'),('outside','^')]:
                part=[r for r in gaps if r['size']==size and r['reference']==ref and r['other_category']==category]
                ax.scatter([int(r['image'][5:]) for r in part],[r['gap_surface_minus_other'] for r in part],label=category,marker=marker)
            for i in NO_LIGHT:ax.axvspan(i-.35,i+.35,color='#ffe2a8',alpha=.5,zorder=0)
            ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(1,22),xlabel='Image number (shaded = light OFF)',ylabel='Mean(surface) - mean(other)',title=f'{size}px | {ref} reference');ax.legend(fontsize=8)
        fig.savefig(OUT/'05_lighting_comparison'/f'paired_gaps_{size}.png',dpi=130);plt.close(fig)
    # Frequency attribution focuses on no-light images, not pooled illumination effects.
    part=[r for r in bands if r['lighting']=='light_off' and r['reference']=='global' and r['other_category'] in ['tag_plain','tag_text']]
    keys=list(dict.fromkeys((r['image'],r['size'],r['other_category']) for r in part))
    if keys:
        matrix=np.array([[next(r['additive_contribution_to_gap'] for r in part if (r['image'],r['size'],r['other_category'])==k and r['band']==b) for b in BANDS] for k in keys])
        fig,ax=plt.subplots(figsize=(9,max(4,len(keys)*.35)),layout='constrained');limit=max(abs(matrix).max(),.01)
        im=ax.imshow(matrix,cmap='coolwarm',vmin=-limit,vmax=limit,aspect='auto');ax.set(xticks=range(4),xticklabels=list(BANDS),yticks=range(len(keys)),yticklabels=[' / '.join(map(str,k)) for k in keys],title='Light OFF: additive contribution to surface-minus-tag cosine gap')
        for (y,x),v in np.ndenumerate(matrix):ax.text(x,y,f'{v:.3f}',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=ax);fig.savefig(OUT/'03_frequency_profiles/no_light_band_contributions.png',dpi=140);plt.close(fig)
    return summary,lighting,gaps,bands


def confirm_extra_reviews():
    data=json.loads((OUT/'roi_definitions.json').read_text(encoding='utf-8'));anns=annotations()
    for r in data['regions']:
        if r['selection']=='manual_replacement':
            assert not any(fft.intersect(r['box'],a['box'])>0 for a in anns[r['image']])
            r['review_status']='assistant_visual_reviewed';r['note']='Actual crop manually inspected by assistant; no obvious defect/label/border; not human-certified'
    unavailable=[]
    for i in range(1,22):
        for category in ['surface','tag_plain','tag_text','outside']:
            for size in [128,256]:
                if not any(r['image']==f'image{i}' and r['category']==category and r['size']==size for r in data['regions']):
                    unavailable.append({'image':f'image{i}','category':category,'size':size,'reason':'No visually verified pure ROI of this category and size'})
    data['unavailable']=unavailable;pc.write_json(OUT/'roi_definitions.json',data)


def reproduce_and_analyze_masks(regions):
    dest=OUT/'06_mask_analysis/reproduction'
    for folder in ['features','references']: (dest/folder).mkdir(parents=True,exist_ok=True)
    anns=annotations();reproduction=[];damage_rows=[];roi_rows=[];stage_rows=[];block_rows=[];all_masks={}
    selected=json.loads((PREVIOUS/'selected_parameters.json').read_text())['selected']
    for i in range(1,22):
        name=f'image{i}';path=SOURCE/(name+'.png')
        if not path.exists():continue
        rgb=pc.read_image(path);shape=rgb.shape[:2]
        rec={'name':name,'path':path,'shape':shape}
        new,status=fm.features(rec,dest)
        old=fm.read_csv(PREVIOUS/'features'/(name+'.csv'))
        assert len(old)==len(new)
        diffs=[]
        for a,b in zip(new,old):
            assert all(a[k]==int(b[k]) for k in ['size','x','y'])
            for k in ['similarity','peak_strength']:
                if b[k]=='':assert a[k] is None
                else:diffs.append(abs(a[k]-float(b[k])))
            assert a['valid_peak']==(b['valid_peak']=='True')
        maxdiff=max(diffs,default=0);assert maxdiff<1e-10
        masks={'none':np.zeros(shape,bool),'legacy128':fm.legacy_mask(shape,new,128),'legacy256':fm.legacy_mask(shape,new,256),
               'proposed':fm.conservative_mask(shape,new,selected['similarity_limit'],selected['peak_limit'])}
        for method,suffix in [('none','none'),('legacy128','legacy128'),('legacy256','legacy'),('proposed','proposed')]:
            oldmask=np.array(Image.open(PREVIOUS/'masks'/f'{name}_{suffix}_excluded.png'))>0
            assert np.array_equal(oldmask,masks[method])
        all_masks[name]=masks
        reproduction.append({'image':name,'blocks':len(new),'max_feature_difference':maxdiff,'all_four_masks_pixel_identical':True})
        pc.montage([pc.draw_boxes(fm.color_mask(rgb,masks[m]),[a['box'] for a in anns[name]]) for m in ['none','legacy128','legacy256','proposed']],
                   ['No exclusion','Legacy 128px','Legacy 256px (main)','Similarity + peaks'],width=512).save(OUT/'06_mask_analysis'/f'{name}_masks.png')
        for size in [128,256]:
            part=[r for r in new if r['size']==size];sims=[r['similarity'] for r in part if r['similarity'] is not None]
            p20=float(np.percentile(sims,20)) if sims else None
            for r in part:
                r['legacy_class']='review' if r['similarity'] is None else 'grid_named_by_legacy' if r['similarity']>=.9 else 'background' if r['similarity']<=p20 else 'review'
            stage_rows.append({'image':name,'lighting':group_for(name),'size':size,'blocks':len(part),'p20':p20,
                'reference_stable':status[str(size)]['stable'],'reference_pairs':status[str(size)]['pairs'],
                'legacy_background_blocks':sum(r['legacy_class']=='background' for r in part),
                'similarity_below_070':sum(r['similarity'] is not None and r['similarity']<.7 for r in part),
                'valid_peak_blocks':sum(r['valid_peak'] for r in part),
                'weak_peak_below_2':sum(r['valid_peak'] and r['peak_strength']<2 for r in part),
                'joint_background_blocks':sum(r['valid_peak'] and r['similarity'] is not None and r['similarity']<.7 and r['peak_strength']<2 for r in part),
                'excluded_area_legacy':float(masks['legacy'+str(size)].mean()),'excluded_area_proposed':float(masks['proposed'].mean())})
        for r in [r for r in regions if r['image']==name]:
            for method,mask in masks.items():
                center,fraction=fm.retention(r['box'],mask)
                roi_rows.append({'roi':r['id'],'image':name,'lighting':group_for(name),'size':r['size'],'category':r['category'],
                    'method':method,'center_kept':center,'retained_fraction':fraction,'label_is_ground_truth':r['human_confirmed']})
        for a in anns[name]:
            for method,mask in masks.items():
                center,fraction=fm.retention(a['box'],mask)
                damage_rows.append({'image':name,'lighting':group_for(name),'annotation_id':a['id'],'label':a['label'],
                                    'method':method,'center_kept':center,'retained_fraction':fraction,'retained_95pct':fraction>=.95})
            cx,cy=(a['box'][0]+a['box'][2])/2,(a['box'][1]+a['box'][3])/2
            covering=[r for r in new if r['x']<=cx<r['x']+r['size'] and r['y']<=cy<r['y']+r['size']]
            for r in covering:
                block_rows.append({'image':name,'annotation_id':a['id'],'label':a['label'],**{k:r[k] for k in ['size','x','y','similarity','peak_strength','valid_peak','legacy_class']}})
            center,area=fm.retention(a['box'],masks['legacy256'])
            if area<.95:
                bad=[r for r in covering if r['size']==256 and r['legacy_class']=='background']
                candidate=min(bad,key=lambda r:r['similarity']) if bad else next((r for r in covering if r['size']==256),None)
                n=256;x=min(max(0,int(cx-n/2)),shape[1]-n);y=min(max(0,int(cy-n/2)),shape[0]-n)
                panels=[pc.draw_boxes(fm.color_mask(rgb,masks[m]),[a['box']])[y:y+n,x:x+n] for m in ['none','legacy128','legacy256','proposed']]
                pc.montage(panels,['Original + GT','Legacy128','Legacy256','Proposed'],width=384,nearest=True).save(OUT/'07_failure_cases'/f'{name}_ann{a["id"]}_context.png')
                if candidate:
                    bx,by=candidate['x'],candidate['y'];gray=fft.read_gray(path);patch=gray[by:by+256,bx:bx+256];_,amp=fft.fft_amplitude(patch)
                    fig,axes=plt.subplots(1,2,figsize=(8,4),layout='constrained');axes[0].imshow(patch,cmap='gray',vmin=0,vmax=255)
                    axes[0].set_title(f'Block ({bx},{by})\nS={candidate["similarity"]:.4f}, class=background');axes[1].imshow(np.log1p(amp),cmap='magma',extent=[-.5,.5,.5,-.5]);axes[1].set(xlim=(-.125,.125),ylim=(.125,-.125),title='Excluded block spectrum (local display scale)')
                    fig.savefig(OUT/'07_failure_cases'/f'{name}_ann{a["id"]}_block_fft.png',dpi=130);plt.close(fig)
        print('Reproduced masks:',name,flush=True)
    pc.csv_out(OUT/'06_mask_analysis/reproduction_checks.csv',reproduction)
    pc.csv_out(OUT/'06_mask_analysis/all_annotation_retention.csv',damage_rows)
    pc.csv_out(OUT/'06_mask_analysis/roi_mask_overlap.csv',roi_rows)
    pc.csv_out(OUT/'06_mask_analysis/decision_stages.csv',stage_rows)
    pc.csv_out(OUT/'07_failure_cases/annotation_center_blocks.csv',block_rows)
    summary=[]
    for group in ['all','light_off','light_on_highres','light_on_lowres']:
        for method in ['none','legacy128','legacy256','proposed']:
            for label in ['all']+list(dict.fromkeys(a['label'] for shapes in anns.values() for a in shapes)):
                part=[r for r in damage_rows if r['method']==method and (group=='all' or r['lighting']==group) and (label=='all' or r['label']==label)]
                if part:summary.append({'lighting':group,'method':method,'label':label,'annotations':len(part),
                    'centers_kept':sum(r['center_kept'] for r in part),'rectangles_95pct_kept':sum(r['retained_95pct'] for r in part),
                    'mean_area_retained':float(np.mean([r['retained_fraction'] for r in part]))})
    pc.csv_out(OUT/'06_mask_analysis/retention_summary.csv',summary)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');cats=['surface','tag_plain','tag_text','outside','damage','tag_edge_mixed']
    for ax,method in zip(axes,['legacy128','legacy256']):
        values=[[1-r['retained_fraction'] for r in roi_rows if r['category']==c and r['method']==method and r['size']==256] for c in cats]
        for j,v in enumerate(values):
            if v:ax.boxplot([v],positions=[j],widths=.5);ax.scatter(np.full(len(v),j),v,s=7,alpha=.4)
        ax.set(xticks=range(len(cats)),xticklabels=cats,ylim=(-.03,1.03),ylabel='Excluded area fraction',title=method+' | 256px ROI');ax.tick_params(axis='x',rotation=30)
    fig.savefig(OUT/'06_mask_analysis/roi_exclusion.png',dpi=140);plt.close(fig)
    return reproduction,damage_rows,summary,stage_rows


def audit_document():
    text='''# 実装監査

|項目|従来コサイン類似度|追加ピーク強度|
|---|---|---|
|入力|元解像度gray、画像全体と128/256pxブロック|同じ画像・ブロック|
|平均除去|窓を掛ける前の単純平均を除去|同じ|
|窓|2次元Hann（外積）|同じ|
|FFT|fft2→fftshift→絶対値|同じ生振幅|
|正規化|振幅をHann窓の総和で割る。log前の単位をそろえる|局所ピーク／環状域の中央値の比|
|比較座標|周波数セル面積で129×129共通座標へ再標本化、各軸±0.125 cycles/pixel|基準周波数を各ブロックのnative FFT binに換算|
|DC|共通座標中心±5点の正方形を除外。中心周波数±0.009765625、セル境界まで約±0.0107421875|基準探索DC半径10 bin、信号円はDCから4 bin超離す等|
|特徴変換|log1p(正規化振幅)|logは使わない|
|基準|画像全体の共通座標振幅スペクトル|画像全体で抽出した最大20点の共役ピーク|
|スコア|dot(Fblock,Fref)/(norm(Fblock)*norm(Fref))|半径3 bin内最大／3–6 bin環状域中央値、参照全点の最大|
|背景判定|S<0.90かつS≤画像内p20|S<0.70かつ強さ<2（前回選択値）|
|重なり|50%重複、格子/背景/保留の多数決。同数は格子→背景→保留|128/256両尺度、全被覆ブロックの一致が必要|
|無効値|類似度計算不能なら保留|参照2方向や比の安定性を満たさなければ保留|

根拠コード: `fft2_cosine_similarity.py` の `fft_amplitude`, `common_spectrum`, `vector`, `cosine`, `classify_similarity_blocks`。
ピーク強度は `point_anomaly_fft_mask.py` の `stable_references`, `peak_stencil`, `prominence`, `conservative_mask`。
`block_peak_matches` が計算する「ピーク位置一致率」と、上記の「ピーク強度比」は別の特徴量。今回の追加方式の再現は強度比を使う。

純粋な一様領域は平均除去後ゼロになり、コサインは未定義。0点や背景と置き換えず欠損として扱う。
正の振幅を比較するコサインは明るさ画像そのものの類似度でも、格子存在の確率でもない。
線形振幅のスケール不変性はlog1p後には厳密には成り立たない。振幅単位・コントラストの差がlog後の形状を変え得る。
振幅だけなので空間位相・位置を捨てており、同じ振幅スペクトルを持つ異なる空間模様を区別できない。
'''
    (OUT/'implementation_audit.md').write_text(text,encoding='utf-8')


def report(regions,features,summary,lighting,gaps,bands,reproduction,retention,retention_summary,stages):
    invalid=[r for r in features if r['similarity_global'] is None]
    report=['# FFT類似度による背景除外の原理検証','','## 1. 目的','',
        '既存FFTコサイン類似度が何に反応するかを、再現・目視ROI・照明条件別比較・周波数帯別寄与・誤除外例から検討する。新しい分類器の開発や有効性の断定は目的としない。','',
        '## 2. 当初の仮説','',
        '格子ライトの周期模様が特徴的なFFTピークを作り、表面と背景を分離するという仮説。ただし従来方式は格子専用のテンプレートではなく画像全体FFTを参照するため、粒状模様、エッジ、ぼけ、照明分布でもスコアが変わる可能性がある。','',
        '## 3. 使用データ','',
        'ライトなし: image1/19/20/21。ライトあり: image2–18（ユーザー申告）。image18は428×321、他は2048×1536。照明あり高解像度16枚・なし4枚・低解像度1枚を別集計する。照明だけを変えた位置・露光・倍率の一致する対画像とは確認できていないので、因果効果は推定しない。',
        f'全傷.jsonの全54矩形を使用。分析ROIは{len(regions)}個、うち傷文脈ブロック{sum(r["category"]=="damage" for r in regions)}個。非傷ROIは座標を手動選択し、実際の切り出しをassistantが目視確認した暫定ラベル。ユーザーによる背景正解マスクではなく、正常を保証もしない。未注釈傷の可能性を残す。',
        '札の角が混入した表面候補はtag_edge_mixed、支持材との境界はoutside_edge_mixed、線状特徴が疑わしい表面はuncertain_surfaceに変更した。これらは正常参照に使わない。変更履歴はroi_definitions.jsonに保存。',
        'ライトなしで純粋な外周背景ROIを得たのはimage19のみで、選択部分は白飛びしている。image1/20/21の暗い外周を、確認なく対象外背景と扱ってはいない。取得不能なカテゴリ・サイズはJSONのunavailableに明記。','',
        '## 4. 実験方法','',
        '[実装監査](implementation_audit.md)に正規化・周波数範囲・窓・判定・投票条件を整理。既存関数で全ブロックの類似度と強度比を再計算し、保存済みCSVと比較。除外なし、従来128/256、新方式のマスクを画素単位で照合した。',
        f'再現画像数{len(reproduction)}。最大数値差={max(r["max_feature_difference"] for r in reproduction):.3g}、4種類のマスクは全画像で一致。元画像・既存コード・成果物は変更せず別フォルダに出力。',
        '実験A: 128/256pxの同一画像・サイズで代表ROIを横並び。Hann総和で正規化した振幅のlog1pを共通カラースケールで表示。半径方向プロファイルは正規化振幅の二乗の環状平均。ピークはDC半径3 bin外の99パーセンタイル以上の局所最大を最大20点表示し、格子由来とは断定しない。',
        '実験B: 全体参照と同一画像・同一サイズの暫定正常表面参照を比較。表面参照は各ROIの共通座標振幅の画素ごとの中央値を作り、その後log1p。評価ROI自身と空間的に重なる参照ROIは除外する。残り参照がない場合は欠損。画像をリサイズしない。',
        '実験C: ROI数で重み付けした全体平均に頼らず、画像別統計と画像平均を等重みで集約。標準偏差は母標準偏差(ddof=0)。1標本では0だが安定性の証拠ではない。重なりROIや同じ傷の複数サイズを独立標本とみなした検定・信頼区間は作らない。',
        '周波数寄与: 従来の比較範囲とDCマスクを固定し、半径0–0.025/0.025–0.050/0.050–0.090/0.090–約0.177 cycles/pixelに分割。帯域ごとのコサイン、全体コサインの加算寄与、帯域を外した際の表面−他領域のスコア差を記録。寄与の和は全体類似度差に一致。高周波端は±0.125の正方形の角なので全方位を含まない。',
        '実験D: 既存の固定条件を再現し、54矩形の中心保持と面積95%以上保持を評価。未確認ROIのマスク重なりは記述統計のみで、背景分類のAccuracyやPrecisionとして報告しない。','',
        '## 5. FFTスペクトルの比較結果','',
        '切り出し位置は01_roi/*_roi_locations.png、領域分類の確認用一覧は*_roi_review.png。全ROIの生振幅・共通座標スペクトルは02_spectra/*.npz、ピーク位置と強度はpeaks.csv。',
        '![照明なしimage21の領域別スペクトル](02_spectra/image21_128_comparison.png)',
        '![照明ありimage8の領域別スペクトル](02_spectra/image8_128_comparison.png)',
        '![照明なしimage20の周波数プロファイル](03_frequency_profiles/image20_128_profiles.png)',
        '周期ピーク、原点付近の集中、広がった粒状成分は異なる特徴。ピーク数は局所最大の個数に過ぎず、格子の検出率ではない。照明なし画像の表面にも周波数の偏りがあり得るため、ピークの存在だけでは格子ライトの有無を判定できない。','',
        '## 6. 類似度の分布','',
        '![128pxの領域別分布](04_similarity/boxplots_128.png)',
        '![256pxの領域別分布](04_similarity/boxplots_256.png)',
        '![128pxヒストグラム](04_similarity/histograms_128.png)',
        f'全体参照で類似度未定義のROIは{len(invalid)}件。ゼロスペクトル等は0点ではなく欠損として除外する。参照別・画像別・カテゴリ別の平均/中央値/標準偏差/最小/最大はsimilarity_summary.csv。',
        'ライトなし画像の同一サイズ内比較（表面平均−札平均。正なら表面の方が基準に似る）:', '',
        '|画像|px|参照|相手|表面平均|相手平均|差|表面n/相手n|','|---|---:|---|---|---:|---:|---:|---:|']
    for r in gaps:
        if r['lighting']=='light_off' and r['other_category'] in ['tag_plain','tag_text','outside']:
            report.append(f"|{r['image']}|{r['size']}|{r['reference']}|{r['other_category']}|{r['surface_mean']:.4f}|{r['other_mean']:.4f}|{r['gap_surface_minus_other']:.4f}|{r['surface_count']}/{r['other_count']}|")
    report+=['','## 7. 格子ライトの有無による違い','',
        '![画像別の類似度差。黄色はライトなし](05_lighting_comparison/paired_gaps_128.png)',
        '![ライトなしの帯域別寄与](03_frequency_profiles/no_light_band_contributions.png)',
        '表の画像間比較は材質・倍率・ピント・露光・位置などを含む観察的比較。image18を高解像度群に混ぜない。帯域のcycles/pixelは物理長さではないので、解像度が異なる画像間で同じ材料スケールとはいえない。','',
        '|照明群|px|参照|カテゴリ|画像数|ROI数|画像平均の平均|画像平均の標準偏差|','|---|---:|---|---|---:|---:|---:|---:|']
    for r in lighting:
        if r['reference']=='global' and r['category'] in ['surface','tag_plain','tag_text','outside']:
            report.append(f"|{r['lighting']}|{r['size']}|{r['reference']}|{r['category']}|{r['count']}|{r['roi_count']}|{r['mean']:.4f}|{r['std']:.4f}|")
    report+=['','## 8. 背景除外の成功・失敗要因','',
        '![image20の全傷と除外マスク](06_mask_analysis/image20_masks.png)',
        '![暫定ROIと除外マスクの重なり](06_mask_analysis/roi_exclusion.png)',
        '|方式|全傷数|中心保持|矩形95%以上保持|平均面積保持|','|---|---:|---:|---:|---:|']
    for r in retention_summary:
        if r['lighting']=='all' and r['label']=='all':report.append(f"|{r['method']}|{r['annotations']}|{r['centers_kept']}|{r['rectangles_95pct_kept']}|{r['mean_area_retained']:.3f}|")
    failed=[r for r in retention if r['method']=='legacy256' and not r['retained_95pct']]
    report+=['','従来256pxで95%保持を満たさなかった矩形:', '', '|画像|注釈ID|ラベル|中心保持|面積保持|','|---|---:|---|---|---:|']
    for r in failed:report.append(f"|{r['image']}|{r['annotation_id']}|{r['label']}|{r['center_kept']}|{r['retained_fraction']:.3f}|")
    if failed:
        r=failed[0];report+=['',f'![誤除外の例](07_failure_cases/{r["image"]}_ann{r["annotation_id"]}_context.png)',
                           f'![対応する背景判定ブロックのFFT](07_failure_cases/{r["image"]}_ann{r["annotation_id"]}_block_fft.png)']
    report+=['','## 9. 考察','',
        '従来方式は画像全体への相対的な似方を測る。撮像された表面が支配的であれば、その広帯域テクスチャと異なる札や外周は低スコアになり得る。逆に、正常表面でも照明むら・粒状模様の密度・局所変形・傷・ブロック内の境界混入により全体から外れる。',
        '下位20%は背景の絶対的な証拠ではない。S≥0.90の優先規則と多数決があるので最終除外面積は20%固定ではないが、低順位を背景と解釈すると少数の本来検査すべき表面も除外し得る。傷の存在がスコアを下げたのか、傷の周囲が元々異なるのかは、無傷の同位置画像がないため切り分けられない。',
        '新方式の除外0%は「全て格子だった」ことを意味しない。参照周波数が128pxでDCに近過ぎる・非平行な2対がそろわない等の参照保留、低類似度条件、強度比条件、両尺度全ブロックの一致が順番に候補を制限する。decision_stages.csvで段階ごとに数を記録する。',
        '札内部、札文字、札縁を一括すると、印刷エッジと滑らかな紙の違いが消える。さらに白飛びした外周はFFTがゼロで、既存の類似度規則では背景ではなく保留となる。',
        '表面参照との比較は全体参照に混ざる札・背景の影響を減らす一方、手動選択した表面の代表性に依存する。参照を変えると類似度分布が変わるので0.90の閾値をそのまま適用できる証拠にはならない。','',
        '## 10. 結論','',
        '**Q1. 主因は格子周期か、一般表面テクスチャか。** このデータでは「格子周期だけ」が主因という説明は成り立たない。ライトなし4画像でも、取得できた札ROIは全体参照・表面参照の双方で表面より低い類似度だった。ライトなしの全体参照では、128pxの表面−札差は画像平均でtag_plain 0.085（3画像すべて正）、tag_text 0.096（2画像すべて正）。256pxでも取得できた3比較はすべて正だった。したがって、表面の粒状性・粗さ、札の滑らかさ、文字・縁の方向性など一般テクスチャの違いが少なくとも一部の分離を説明する。ライトあり表面では周期的な空間模様とスペクトル集中が明瞭な例があり、格子は追加の識別情報になり得るが、同一対象のON/OFF対画像がないので寄与率は分離できない。','',
        '**Q2. ライトなし画像でも差があるか。** 札については差がある。全体参照の表面平均−札平均はimage1=0.093（128px）、image19=0.045/0.055（128/256px）、image20=0.080/0.097、image21=0.083～0.176で、観測した全比較が正だった。表面参照でも同じ符号だった。ただしサンプル数は少なく、ROIは暫定ラベルである。純粋な外周背景はimage1/20/21では取得できず、image19の候補は白飛びしてコサイン未定義だったため、「ライトなしでも外周背景が明確に分離する」とは結論できない。','',
        '**Q3. 寄与する周波数帯。** ライトなしの表面−札差に対する加算寄与は、全体参照128pxで0.05–0.09が平均+0.054、0.09–約0.177 cycles/pixelが+0.120。256pxではそれぞれ+0.085、+0.109だった。0–0.025は−0.062/−0.069で、低周波帯だけなら札の方が全体参照に似る方向だった。従ってライトなしでの札分離は主として中高周波の粒状成分・細かなエッジ形状に対応する。ただしこれはコサイン差の帯域分解であって、物理的な因果や個別ピークの格子由来を証明しない。','',
        '**Q4. 正常表面や傷が誤除外される理由。** 従来方式は絶対的な背景モデルではなく、画像内で低い20%を背景候補にする。局所的な照明むら、模様密度、ぼけ、傷、札・外周との境界混入によって全体FFTから外れれば表面でも低順位になる。256px従来方式では全54矩形中、中心保持50件、矩形面積95%以上保持49件。失敗はimage14の線状異常1件（89.3%保持）と、image16/18/20の点状異常4件（0%保持）。image20の例は傷そのものだけでなく札縁を含む256pxブロックで、類似度0.6768だった。傷が原因か周囲構成が原因かは、同位置の無傷対画像がなく分離できない。','',
        '**Q5. 一般的な検査対象領域分類に使えるか。** 候補特徴としては使えそうだが、単独での実用分類器とはまだいえない。ライトなしでも札との一貫した差があり、正常表面参照に変えても差が維持・拡大する例があるため、格子専用特徴ではない。一方、外周背景の確定ROI不足、下位20%の相対判定、傷誤除外、解像度依存が残る。「類似度0.90以上=格子」ではなく「画像全体または代表表面の周波数テクスチャに近い」と解釈するのが妥当。背景・札・表面の確定ROIを用意した後で初めて分類性能を測れる。','',
        '**総合判断。** 観測結果は、ライトありでは周期模様、ライトなしでは主に中高周波の表面テクスチャ差が同じコサイン類似度へ寄与する、という複合説明と整合する。現データだけでどちらか一方を唯一の原因とは断定しない。ピーク強度方式が0%除外になったのも、背景が存在しないからではなく、128/256両尺度の安定参照・低類似度・弱ピーク・全被覆合意という保守条件がほぼ成立しなかったためである。','',
        '## 11. 次の実験方針','',
        '1. ROI一覧をユーザーが確認・修正し、札・表面・支持材の少数の確定ROIを用意する。今回の自動集計を背景分類の正解率と混同しない。',
        '2. 同一対象・同一視野・同一解像度・同一露光条件でライトON/OFFを対にして撮像する。これが周期性と一般テクスチャの寄与を区別する直接的な比較になる。',
        '3. 対画像で特徴周波数だけがON時に強まるかを確認し、同じROIで帯域除外前後の類似度差を比較する。現在の帯域分析は関連を示すもので原因を確定するものではない。',
        '4. 背景の正解がそろった後に、表面保持率・札除外率・外周除外率を分けて固定条件で評価する。現時点では新しい複雑な分類器は追加しない。','',
        '再実行: `python fft_region_analysis.py --run`。ROI編集後は `--preview` で表示し、確認済みROIのみ `review_status=human_confirmed` と `human_confirmed=true` にする。`--prepare` は初期候補JSONを作り直すため、既存JSONがある場合は停止する。']
    (OUT/'report.md').write_text('\n\n'.join(report).replace('|\n\n|','|\n|')+'\n',encoding='utf-8')


def run():
    setup();audit_document()
    paths=[SOURCE/'全傷.json']+list(SOURCE.glob('image*.png'))+[ROOT/'fft2_cosine_similarity.py',ROOT/'point_anomaly_fft_mask.py',ROOT/'point_anomaly_contrast.py']
    paths+=[p for directory in [PREVIOUS,ROOT/'outputs/point_anomaly_contrast'] for p in directory.rglob('*') if p.is_file()]
    hashes={str(p.resolve()):pc.digest(p) for p in paths}
    data=json.loads((OUT/'roi_definitions.json').read_text(encoding='utf-8'));regions=data['regions']
    assert not any(r['review_status']=='pending_visual_crop_review' for r in regions)
    reproduction,retention,rs,stages=reproduce_and_analyze_masks(regions)
    features,cache=compute_roi_features(regions)
    spectrum_figures(regions,features,cache)
    summary,lighting,gaps,bands=summarize_similarity(features)
    report(regions,features,summary,lighting,gaps,bands,reproduction,retention,rs,stages)
    assert all(pc.digest(Path(p))==h for p,h in hashes.items()),'Input modified'
    pc.write_json(OUT/'verification.json',{'unchanged_input_files':len(hashes),'input_hashes_unchanged':True,
        'reproduced_images':len(reproduction),'all_masks_pixel_identical':True,'roi_count':len(regions),
        'annotations_evaluated':len(retention)//4,'band_contributions_sum_to_full_cosine':True})
    pc.write_json(OUT/'run_manifest.json',{'source_hashes':hashes,'roi_sha256':pc.digest(OUT/'roi_definitions.json'),
        'script_sha256':pc.digest(Path(__file__)),'versions':{'numpy':np.__version__,'opencv':cv2.__version__,'matplotlib':matplotlib.__version__}})
    print('Analysis complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    if args.prepare:
        if (OUT/'roi_definitions.json').exists():raise FileExistsError('ROI definitions already exist; edit the JSON explicitly')
        generate_rois()
    elif args.run:run()
    elif args.preview:roi_previews(json.loads((OUT/'roi_definitions.json').read_text(encoding='utf-8'))['regions'])
    else:overview()


