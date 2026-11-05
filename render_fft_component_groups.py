"""Render separate inverse-FFT images for base, harmonic, mixed, and all groups."""
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data20260903'/'data20260903';OUT=ROOT/'analysis_fft_component_extraction'
def mask(shape, rows, radius=2):
 y,x=np.ogrid[:shape[0],:shape[1]];m=np.zeros(shape,bool)
 for r in rows.itertuples():
  for yy,xx in ((r.y,r.x),((-r.y)%shape[0],(-r.x)%shape[1])):m|=(y-yy)**2+(x-xx)**2<=radius*radius
 return m
def main():
 s=pd.read_csv(OUT/'selected_frequency_components.csv')
 for n in (7,8,11,12,18):
  img=cv2.imread(str(DATA/f'image{n}.png'),cv2.IMREAD_GRAYSCALE).astype(float);F=np.fft.fft2(img-img.mean());lim=np.percentile(img,[1,99])
  for method in ('legacy','lattice'):
   q=s[(s.image==n)&(s.method==method)&(s.condition=='all')].drop_duplicates(['component_type','peak_id'])
   groups=[('base',q[q.component_type=='base']),('harmonic',q[q.component_type=='harmonic']),('mixed',q[q.component_type=='mixed']),('all',q)]
   ims=[]
   for _,g in groups:ims.append(np.fft.ifft2(F*mask(img.shape,g)).real if len(g) else np.zeros_like(img))
   scale=max(np.percentile(abs(a),99) for a in ims) or 1
   fig,ax=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
   ax[0,0].imshow(img[::3,::3],cmap='gray',vmin=lim[0],vmax=lim[1]);ax[0,0].set_title('Original')
   for a,(name,_),im in zip(ax.ravel()[1:],groups,ims):a.imshow(im[::3,::3],cmap='coolwarm',vmin=-scale,vmax=scale);a.set_title(f'Extracted: {name}')
   ax[1,2].imshow((img-ims[-1])[::3,::3],cmap='gray',vmin=lim[0],vmax=lim[1]);ax[1,2].set_title('Residual after all')
   for a in ax.ravel():a.set_xticks([]);a.set_yticks([])
   fig.suptitle(f'image{n}: {method} components; radius 2 bins');fig.savefig(OUT/f'image{n:02d}'/f'component_groups_{method}.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
