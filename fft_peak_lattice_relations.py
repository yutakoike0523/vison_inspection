"""Global Hann-FFT peak-lattice analysis; no filtering or detector evaluation."""
from __future__ import annotations
import csv, math
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data20260903'/'data20260903';OUT=ROOT/'analysis_fft_peak_lattice_relations';PRIMARY={7,8,11,12,18};ALL=list(range(2,19));EPS=1e-12
def save(p,rows):
 k=list(dict.fromkeys(x for r in rows for x in r));
 with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=k,extrasaction='ignore');w.writeheader();w.writerows(rows)
def fftpeaks(n):
 import cv2
 a=cv2.imread(str(DATA/f'image{n}.png'),cv2.IMREAD_GRAYSCALE).astype(float);h,w=a.shape;z=np.fft.fftshift(np.fft.fft2((a-a.mean())*np.outer(np.hanning(h),np.hanning(w))));p=abs(z)**2;fy=np.fft.fftshift(np.fft.fftfreq(h));fx=np.fft.fftshift(np.fft.fftfreq(w));cy,cx=h//2,w//2;loc=np.log10(p+EPS)==maximum_filter(np.log10(p+EPS),5); rows=[]
 for y,x in zip(*np.nonzero(loc)):
  if math.hypot(y-cy,x-cx)<=4 or not(fy[y]>0 or(abs(fy[y])<1e-12 and fx[x]>0)):continue
  sy,sx=(-y)%h,(-x)%w; cut=p[max(0,y-8):min(h,y+9),max(0,x-8):min(w,x+9)]; prom=10*np.log10(p[y,x]/max(np.median(cut),EPS));d=math.hypot(fx[x],fy[y]);rows.append({'image':n,'peak_id':0,'y':y,'x':x,'sym_y':sy,'sym_x':sx,'fx':fx[x],'fy':fy[y],'bin_dx':x-cx,'bin_dy':y-cy,'frequency':d,'period_px':1/d,'angle_deg':math.degrees(math.atan2(fy[y],fx[x])),'power':(p[y,x]+p[sy,sx])/2,'power_db':10*np.log10((p[y,x]+p[sy,sx])/2+EPS),'prominence_db':prom})
 rows.sort(key=lambda r:r['power'],reverse=True)
 for i,r in enumerate(rows,1):r['peak_id']=i
 return a,z,rows
def canon(v):return v[1]>0 or(abs(v[1])<1e-12 and v[0]>0)
def match(b1,b2,obs,h,w):
 pred=[]
 for m in range(-4,5):
  for n in range(-4,5):
   if (m,n)==(0,0) or (abs(m)+abs(n)==1):continue
   v=(m*b1['fx']+n*b2['fx'],m*b1['fy']+n*b2['fy'])
   if math.hypot(*v)>=.48 or not canon(v):continue
   typ='single_harmonic' if m==0 or n==0 else 'mixed'; label=(f'{abs(m)}g1' if n==0 else f'{abs(n)}g2' if m==0 else f'{m}g1{n:+}g2')
   pred.append((v,m,n,typ,label,abs(m)+abs(n)))
 # include bases
 pred += [((b1['fx'],b1['fy']),1,0,'fundamental','g1',1),((b2['fx'],b2['fy']),0,1,'fundamental','g2',1)]
 C=np.full((len(pred),len(obs)),99.);tol=2
 for i,(v,*_) in enumerate(pred):
  for j,o in enumerate(obs):C[i,j]=math.hypot((v[0]-o['fx'])*w,(v[1]-o['fy'])*h)
 ii,jj=linear_sum_assignment(C);out=[]
 for i,j in zip(ii,jj):
  if C[i,j]<=tol:
   v,m,n,t,l,order=pred[i];o=obs[j];out.append({'observed_peak_id':o['peak_id'],'pred_fx':v[0],'pred_fy':v[1],'m':m,'n':n,'classification':t,'label':l,'order':order,'error_bins':C[i,j],'error_cpp':math.hypot(v[0]-o['fx'],v[1]-o['fy']),**o})
 return out
def main():
 OUT.mkdir(exist_ok=True);peaks=[];cand=[];matches=[];unexp=[]
 for n in ALL:
  print(n,flush=True);img,z,rows=fftpeaks(n);peaks+=rows[:30];obs=rows[:30];bases=[r for r in rows[:10] if r['period_px']>=35]
  opts=[]
  for i,a in enumerate(bases):
   for b in bases[i+1:]:
    dot=abs(a['fx']*b['fx']+a['fy']*b['fy'])/(a['frequency']*b['frequency']);ang=math.degrees(math.acos(min(1,dot)))
    if ang<20:continue
    q=match(a,b,obs, img.shape[0],img.shape[1]);score=sum(x['power_db'] for x in q)-8*sum(x['order']-1 for x in q)-3*np.mean([x['error_bins'] for x in q]) if q else -999
    opts.append((score,a,b,q,ang))
  opts.sort(key=lambda x:x[0],reverse=True)
  for rank,(score,a,b,q,ang) in enumerate(opts,1):cand.append({'image':n,'candidate_rank':rank,'g1_peak_id':a['peak_id'],'g2_peak_id':b['peak_id'],'g1_fx':a['fx'],'g1_fy':a['fy'],'g2_fx':b['fx'],'g2_fy':b['fy'],'angle_separation_deg':ang,'explained_top10':sum(x['observed_peak_id']<=10 for x in q),'explained_top30':len(q),'mean_error_bins':np.mean([x['error_bins'] for x in q]) if q else math.nan,'score':score,'is_legacy_top2':a['peak_id']==1 and b['peak_id']==2})
  if not opts:continue
  best=opts[0];used={x['observed_peak_id'] for x in best[3]}
  for x in best[3]:matches.append({'image':n,'candidate_rank':1,**x})
  for o in obs:
   if o['peak_id'] not in used:unexp.append({'image':n,**o})
  if n in PRIMARY:
   fig,ax=plt.subplots(1,2,figsize=(13,6),constrained_layout=True);db=np.log10(abs(z)**2+EPS);lo,hi=np.percentile(db,[10,99.8])
   for k,(title,base) in enumerate((('legacy top-2', (rows[0],rows[1])),('lattice-ranked pair',(best[1],best[2])))):
    ax[k].imshow(db,cmap='magma',vmin=lo,vmax=hi);ax[k].set_title(f'image{n}: {title}')
    q=match(*base,obs,img.shape[0],img.shape[1]);used={x['observed_peak_id']:x for x in q}
    for o in obs:
     x=used.get(o['peak_id']);col='red' if x is None else ('blue' if x['classification']=='fundamental' else 'green' if x['classification']=='single_harmonic' else 'orange');ax[k].plot([o['x'],o['sym_x']],[o['y'],o['sym_y']],marker='o',ls='',color=col,ms=4)
     if x:ax[k].text(o['x']+3,o['y']+3,x['label'],color=col,fontsize=6)
    ax[k].set_xlim(img.shape[1]//2-180,img.shape[1]//2+180);ax[k].set_ylim(img.shape[0]//2+180,img.shape[0]//2-180)
   fig.savefig(OUT/f'image{n:02d}_peak_lattice.png',dpi=160);plt.close(fig)
 save(OUT/'fft_peak_pairs.csv',peaks);save(OUT/'fundamental_pair_candidates.csv',cand);save(OUT/'peak_matching_results.csv',matches);save(OUT/'unexplained_peaks.csv',unexp);save(OUT/'fundamental_pair_ranking.csv',[r for r in cand if r['candidate_rank']<=10]);save(OUT/'predicted_harmonics.csv',matches)
 (OUT/'report_ja.md').write_text('# FFTピーク格子関係分析\n\n基本波候補は上位10ピークから選び、角度差20°以上、周期35px以上を満たす非共線ペアを比較した。整数係数は-4..4、実測との対応は2FFT-bin以内かつHungarian法で一対一に制約した。scoreは一致ピーク強度、低次係数、座標誤差を併用した探索用順位であり、格子由来の確定判定ではない。\n\n青=基本波、緑=単一方向高調波、橙=混合成分、赤=未説明。詳細はCSVを参照。\n',encoding='utf-8')
if __name__=='__main__':main()
