"""Export reference spectra and inspect changed reference selections."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import evaluate_all_radial_fft as prev
import multiscale_fft_experiment as base
import stride_grid_overlap_experiment as old
import reference_selection_analysis as ra

root=Path('output_low_frequency_analysis'); summary=root/'summary'
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def save(path,rows): prev.safe_csv(path,rows)
rep=read(summary/'representative_spectra.csv'); details=read(root/'reference_selection'/'reference_selection_details.csv'); spectra=[]; comparisons=[]
for img,inst,position in sorted({(int(x['image']),int(x['instance']),x['position']) for x in rep}):
    q=[x for x in rep if int(x['image'])==img and int(x['instance'])==inst and x['position']==position]; x,y=int(q[0]['x']),int(q[0]['y']); r=old.load_result(Path('output_all_images_radial_fft')/f'image{img}'/'result.npz'); i=int(np.flatnonzero(r['xs']==x)[0]); j=int(np.flatnonzero(r['ys']==y)[0]); bits=[int(v)-1 for v in q[0]['reference_ids'].split()]
    for k,row in enumerate(q):
        z=dict(image=img,instance=inst,position=position,x=x,y=y,frequency=float(row['frequency']),target_db=float(row['target_db']),median_db=float(row['median_db']),difference_db=float(row['difference_db']),difference_squared=float(row['difference_squared']))
        for direction in bits:
            dx,dy=ra.OFF[direction]; xx=int(np.flatnonzero(r['xs']==x+dx*128)[0]); yy=int(np.flatnonzero(r['ys']==y+dy*128)[0]); z[f'reference_{direction+1}_db']=float(r['spectra'][yy,xx,k])
        spectra.append(z)
    if position!='low_max': continue
    image=np.asarray(Image.open(base.DATA/f'image{img}.png').convert('L')); labels=prev.annotation(base.DATA/f'image{img}.json',image.shape); mask=labels[inst][1]; ds=[z for z in details if int(z['image'])==img and int(z['instance'])==inst]; fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    for ax,key,title in zip(axes,['fixed_selected','low_reselected'],['Fixed full-spectrum selection','Low-frequency reselection']):
        ax.imshow(image,cmap='gray',vmin=0,vmax=255); ax.contour(mask,levels=[.5],colors='cyan'); ax.add_patch(Rectangle((x-128,y-128),256,256,fill=False,edgecolor='red',lw=2))
        for z in ds:
            if z[key]!='True': continue
            xp,yp=int(z['reference_x']),int(z['reference_y']); ax.add_patch(Rectangle((xp-128,yp-128),256,256,fill=False,edgecolor='yellow')); ax.text(xp,yp,z['reference_id'],color='yellow')
        ax.set(xlim=(max(0,x-280),min(image.shape[1],x+280)),ylim=(min(image.shape[0],y+280),max(0,y-280)),title=title)
    fig.savefig(root/'reference_selection'/'selected_reference_figures'/f'image{img}_defect{inst}.png',dpi=120); plt.close(fig)
    subset=np.flatnonzero((np.flatnonzero(r['bin_valid'])+.5)*.01<.05)
    for method,key in [('fixed','fixed_selected'),('reselected','low_reselected')]:
        use=[z for z in ds if z[key]=='True']; arr=[]
        for z in use:
            xp,yp=int(z['reference_x']),int(z['reference_y']); ix=int(np.flatnonzero(r['xs']==xp)[0]); iy=int(np.flatnonzero(r['ys']==yp)[0]); arr.append(r['spectra'][iy,ix])
        a=np.stack(arr); dis=[np.sqrt(np.mean((a[v,subset]-a[w,subset])**2)) for v in range(len(a)) for w in range(v+1,len(a))]
        comparisons.append(dict(image=img,instance=inst,method=method,count=len(use),mean_pairwise_low_rms=float(np.mean(dis)) if dis else None,mean_background_fraction=float(np.mean([float(z['background_fraction']) for z in use])),mean_tag_fraction=float(np.mean([float(z['tag_fraction']) for z in use])),mean_deformation_fraction=float(np.mean([float(z['deformation_fraction']) for z in use])),ids=' '.join(z['reference_id'] for z in use)))
save(summary/'representative_reference_spectra.csv',spectra); save(root/'reference_selection'/'representative_selection_comparison.csv',comparisons)
