"""Reconstruct selected peak components and residuals; does not run a detector."""
from __future__ import annotations
import csv, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fft_peak_lattice_relations as lattice
import fft_grid_notch_removal_analysis as masks

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data20260903'/'data20260903';SRC=ROOT/'analysis_fft_peak_lattice_relations';OUT=ROOT/'analysis_fft_component_extraction';ALL=list(range(2,19));PRIMARY={7,8,11,12,18};EPS=1e-12
def read(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,rows):
 k=list(dict.fromkeys(x for r in rows for x in r));
 with open(p,'w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=k,extrasaction='ignore');w.writeheader();w.writerows(rows)
def ints(r):
 if 'image' in r:r['image']=int(float(r['image']))
 for k in ('peak_id','y','x','sym_y','sym_x','g1_peak_id','g2_peak_id','candidate_rank'):
  if k in r:r[k]=int(float(r[k]))
 for k in ('fx','fy','power','power_db','frequency','period_px','angle_deg'):
  if k in r:r[k]=float(r[k])
 return r
def disk(shape,rows,r):
 y,x=np.ogrid[:shape[0],:shape[1]];m=np.zeros(shape,bool)
 for q in rows:
  for yy,xx in ((q['y'],q['x']),(q['sym_y'],q['sym_x'])):m|=(y-yy)**2+(x-xx)**2<=r*r
 return m
def comp(groups,kind):
 b=groups['base'];h=groups['harmonic'];m=groups['mixed']
 return b if kind=='base' else b+h if kind=='harmonics' else b+m if kind=='mixed' else b+h+m
def plot(img,g,r,mask,rows,title,path):
 lim=np.percentile(img,[1,99]);gl=max(np.percentile(abs(g),99),1);fig,ax=plt.subplots(1,4,figsize=(18,5),constrained_layout=True)
 for a,aimg,ttl,cmap in zip(ax,(img,g,r,g),('Original','Extracted FFT components','Residual (Original - Extracted)','Removed components (signed)'),('gray','coolwarm','gray','coolwarm')):
  if cmap=='gray':a.imshow(aimg[::3,::3],cmap=cmap,vmin=lim[0],vmax=lim[1])
  else:a.imshow(aimg[::3,::3],cmap=cmap,vmin=-gl,vmax=gl)
  a.set_title(ttl);a.set_xticks([]);a.set_yticks([])
 fig.suptitle(title);fig.savefig(path,dpi=120);plt.close(fig)
def fftplot(F,groups,shape,path):
 db=np.log10(abs(np.fft.fftshift(F))**2+EPS);lo,hi=np.percentile(db,[8,99.8]);fig,ax=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
 for a,title in zip(ax,('Selected peaks','Radius-2 extraction mask')):a.imshow(db if title[0]=='S' else np.zeros(shape),cmap='magma',vmin=lo,vmax=hi);a.set_title(title)
 for typ,col in [('base','blue'),('harmonic','lime'),('mixed','orange')]:
  for q in groups[typ]:
   for y,x in ((q['y'],q['x']),(q['sym_y'],q['sym_x'])):ax[0].plot(x,y,'o',ms=4,color=col);ax[1].plot(x,y,'o',ms=4,color=col)
 for a in ax:a.set_xlim(shape[1]//2-180,shape[1]//2+180);a.set_ylim(shape[0]//2+180,shape[0]//2-180)
 fig.savefig(path,dpi=150);plt.close(fig)
def main():
 OUT.mkdir(exist_ok=True);peaks=[ints(x) for x in read(SRC/'fft_peak_pairs.csv')];cand=[ints(x) for x in read(SRC/'fundamental_pair_candidates.csv')];match=[ints(x) for x in read(SRC/'peak_matching_results.csv')];summ=[];selected=[];validation=[];defrows=[]
 for n in ALL:
  import cv2
  img=cv2.imread(str(DATA/f'image{n}.png'),cv2.IMREAD_GRAYSCALE).astype(float);h,w=img.shape;F=np.fft.fft2(img-img.mean());pr=sorted([x for x in peaks if x['image']==n],key=lambda q:q['peak_id']);obs=pr[:30];legacy=(obs[0],obs[1]);best=next((x for x in cand if x['image']==n and x['candidate_rank']==1),None);ranked=legacy if best is None else (next((x for x in obs if x['peak_id']==best['g1_peak_id']),obs[0]),next((x for x in obs if x['peak_id']==best['g2_peak_id']),obs[1]))
  methods={'legacy':legacy,'lattice':ranked};idir=OUT/f'image{n:02d}';idir.mkdir(exist_ok=True);d,e,grid,defs=masks.mask_from_annotations(n,img.shape);valid=~(d|e)
  for method,bases in methods.items():
   # use the same low-order matching rule as the preceding lattice analysis.
   mt=lattice.match(*bases,obs,h,w);groups={'base':list(bases),'harmonic':[],'mixed':[]}
   for q in mt:
    if q['classification']=='single_harmonic':groups['harmonic'].append(q)
    elif q['classification']=='mixed':groups['mixed'].append(q)
   # de-duplicate observed coordinates while preserving actual measured peak coordinates.
   for k in groups:
    u={q['peak_id']:q for q in groups[k]};groups[k]=list(u.values())
   if n in PRIMARY:fftplot(F,groups,img.shape,idir/f'fft_{method}_selected_peaks.png')
   for kind in ('base','harmonics','mixed','all'):
    rows=comp(groups,kind);m=disk(img.shape,rows,2);G=np.fft.ifft2(F*m).real;R=img-G;err=float(np.max(abs(G+R-img)));en=float(np.sum(G*G)/max(np.sum((img-img.mean())**2),EPS));tag=f'{method}_{kind}'
    selected += [{'image':n,'method':method,'condition':kind,'component_type':k,'peak_id':q['peak_id'],'fx':q['fx'],'fy':q['fy'],'y':q['y'],'x':q['x'],'radius_bins':2} for k in groups for q in (groups[k] if k=='base' or kind!='base' else [])]
    row={'image':n,'method':method,'condition':kind,'selected_peak_pairs':len(rows),'selected_fft_bins':int(m.sum()),'extracted_rms':float(np.std(G)),'extracted_min':float(G.min()),'extracted_max':float(G.max()),'energy_ratio_to_centered_original':en,'residual_rms':float(np.std(R)),'difference_rms':float(np.sqrt(np.mean((img-R)**2))),'reconstruction_max_abs_error':err,'defect_extracted_rms':float(np.std(G[d])) if d.any() else math.nan,'normal_extracted_rms':float(np.std(G[valid])),'defect_residual_contrast':abs(float(R[d].mean()-R[valid].mean())) if d.any() else math.nan,'normal_residual_std':float(np.std(R[valid]))};summ.append(row);validation.append({'image':n,'method':method,'condition':kind,'max_abs_error':err,'identity_holds':err<1e-9})
    if n in PRIMARY:plot(img,G,R,m,rows,f'image{n} {tag}',idir/f'comparison_{tag}.png');np.savez_compressed(idir/f'float_{tag}.npz',original=img.astype(np.float32),extracted=G.astype(np.float32),residual=R.astype(np.float32),mask=m)
    for j,a in enumerate(defs,1):defrows.append({'image':n,'defect_number':j,'method':method,'condition':kind,'extracted_rms':float(np.std(G[a['mask']])),'residual_mean':float(R[a['mask']].mean()),'residual_std':float(R[a['mask']].std())})
 save(OUT/'extraction_summary.csv',summ);save(OUT/'selected_frequency_components.csv',selected);save(OUT/'reconstruction_validation.csv',validation);save(OUT/'defect_region_comparison.csv',defrows)
 (OUT/'report_ja.md').write_text('# FFT成分抽出と残差比較\n\nHann FFTはピーク位置の特定だけに使い、抽出は同じ平均除去・非Hann元画像の複素FFTで行った。従って各条件で Extracted + Residual = Original を浮動小数点で検証した。\n\n基本波は候補ペア、追加成分は一対一対応済みの実測ピークだけを採用した。抽出成分は格子だけでなく同一周波数の傷・照明成分も含むため、RMSだけで格子除去成功とは判断しない。\n',encoding='utf-8')
if __name__=='__main__':main()
