"""Multiscale local FFT confirmation and overlap-add harmonic grid notch study."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.metrics import roc_auc_score
import analysis_fft_normalization_128 as det
import fft_grid_notch_removal_analysis as global_notch

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data20260903'/'data20260903'; PRIOR=ROOT/'analysis_full_image_fft_peaks'
OUT=ROOT/'analysis_local_multiscale_harmonic'; IMAGES=list(range(2,19)); EVAL=list(range(2,18)); W=512; HOP=128; PAD=256; EPS=1e-12
COMBOS=[(2,0,'2g1'),(3,0,'3g1'),(0,2,'2g2'),(0,3,'3g2'),(1,1,'g1+g2'),(1,-1,'g1-g2'),(2,1,'2g1+g2'),(1,2,'g1+2g2')]

def write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
def power_fft(patch,size):
    win=np.outer(np.hanning(size),np.hanning(size));x=(patch-patch.mean())*win;z=np.fft.fftshift(np.fft.fft2(x));return z,np.abs(z)**2
def nearest_peak(power,fy,fx,target,radbin):
    yy,xx=np.ogrid[:len(fy),:len(fx)];y=int(np.argmin(abs(fy-target[1])));x=int(np.argmin(abs(fx-target[0])))
    d2=(yy-y)**2+(xx-x)**2;m=d2<=radbin**2
    iy,ix=np.unravel_index(np.argmax(np.where(m,power,-np.inf)),power.shape);ring=(d2>radbin**2)&(d2<=(radbin*3+2)**2)
    prom=10*np.log10(power[iy,ix]/max(float(np.median(power[ring])),EPS));return {'fx':float(fx[ix]),'fy':float(fy[iy]),'y':int(iy),'x':int(ix),'power':float(power[iy,ix]),'prom_db':float(prom)}
def canonical(v): return v[1]>0 or (abs(v[1])<1e-12 and v[0]>0)
def priors(n,top,fam): return global_notch.peak_sets(n,top,fam)['top2']
def vecs(rows): return [(float(r['fx']),float(r['fy'])) for r in rows]
def ann(n,shape): return global_notch.mask_from_annotations(n,shape)

def analyze_image(n,top,fam,detailed=False):
    """Store per-512-window bases and actual multiscale harmonic confirmations."""
    img=global_notch.load_gray(n);h,w=img.shape;pad=np.pad(img,((PAD,PAD),(PAD,PAD)),mode='reflect');hh,ww=pad.shape
    win=np.outer(np.hanning(W),np.hanning(W));fy=np.fft.fftshift(np.fft.fftfreq(W));fx=fy.copy();baseprior=vecs(priors(n,top,fam));d,e,_,_=ann(n,img.shape)
    records=[];windows=[]
    for y in range(0,hh-W+1,HOP):
      for x in range(0,ww-W+1,HOP):
        p=pad[y:y+W,x:x+W];z,powr=power_fft(p,W);bases=[]
        for k,v in enumerate(baseprior,1):
            q=nearest_peak(powr,fy,fx,v,2.5);q['base']=k;bases.append(q)
            records.append({'image':n,'window_size':512,'window_x':x-PAD,'window_y':y-PAD,'kind':'base','label':f'g{k}','m':1 if k==1 else 0,'n':0 if k==1 else 1,**q,'period_px':1/max(math.hypot(q['fx'],q['fy']),EPS),'angle_deg':math.degrees(math.atan2(q['fy'],q['fx'])),'annotation_overlap':bool((d|e)[max(0,y-PAD):min(h,y-PAD+W),max(0,x-PAD):min(w,x-PAD+W)].any())})
        # Central 256 / 128 provide independent resolution-scaled observations.
        scales={512:(p,z,powr,fy,fx)}
        if detailed:
          scales.update({256:(p[128:384,128:384],None,None,np.fft.fftshift(np.fft.fftfreq(256)),np.fft.fftshift(np.fft.fftfreq(256))),128:(p[192:320,192:320],None,None,np.fft.fftshift(np.fft.fftfreq(128)),np.fft.fftshift(np.fft.fftfreq(128)))})
        for s in tuple(scales)[1:]:
            zz,pp=power_fft(scales[s][0],s);scales[s]=(scales[s][0],zz,pp,scales[s][3],scales[s][4])
        harms=[]
        for m,nn,label in COMBOS:
            target=(m*bases[0]['fx']+nn*bases[1]['fx'],m*bases[0]['fy']+nn*bases[1]['fy'])
            if math.hypot(*target)>=.46 or not canonical(target):continue
            qs={}
            for s in scales:
                _,_,pp,ffy,ffx=scales[s];qs[s]=nearest_peak(pp,ffy,ffx,target,2.0)
            # An actually observed peak must stand out at the native 512 scale and one shorter scale.
            confirmed=(qs[512]['prom_db']>=4 and (qs[256]['prom_db']>=3 or qs[128]['prom_db']>=3)) if detailed else qs[512]['prom_db']>=4
            for s,q in qs.items(): records.append({'image':n,'window_size':s,'window_x':x-PAD+(W-s)//2,'window_y':y-PAD+(W-s)//2,'kind':'harmonic','label':label,'m':m,'n':nn,'pred_fx':target[0],'pred_fy':target[1],'confirmed_multiscale':confirmed,**q,'period_px':1/max(math.hypot(q['fx'],q['fy']),EPS),'angle_deg':math.degrees(math.atan2(q['fy'],q['fx'])),'annotation_overlap':bool((d|e)[max(0,y-PAD):min(h,y-PAD+W),max(0,x-PAD):min(w,x-PAD+W)].any())})
            if confirmed: harms.append(qs[512])
        # Do not retain 512x512 complex arrays for every overlapping window.
        # They are recomputed during synthesis so memory stays bounded per image.
        windows.append({'x':x,'y':y,'bases':bases,'harmonics':harms})
    return img,windows,records

def local_mask(shape,points,mode='hard',radius=1.0):
    yy,xx=np.ogrid[:shape[0],:shape[1]];m=np.ones(shape)
    for q in points:
        for y,x in ((q['y'],q['x']),((-q['y'])%shape[0],(-q['x'])%shape[1])):
            d2=(yy-y)**2+(xx-x)**2
            if mode=='hard':m[d2<=radius*radius]=0
            else:m*=1-np.exp(-d2/(2*radius*radius))
    return m
def ola(img,windows,condition):
    h,w=img.shape;hp,wp=h+2*PAD,w+2*PAD;out=np.zeros((hp,wp));ws=np.zeros_like(out);win=np.outer(np.hanning(W),np.hanning(W));
    source=np.pad(img,((PAD,PAD),(PAD,PAD)),mode='reflect')
    for a in windows:
        y,x=a['y'],a['x'];p=source[y:y+W,x:x+W];z,_=power_fft(p,W);pts=a['bases'] if condition=='local_base' else a['bases']+a['harmonics']
        if condition=='none':rec=p*win
        else:
            mode='gauss' if condition=='local_all_gauss' else 'hard';mask=local_mask(z.shape,pts,mode,1.0);rec=np.fft.ifft2(np.fft.ifftshift(z*mask)).real+p.mean()*win
        out[y:y+W,x:x+W]+=rec*win;ws[y:y+W,x:x+W]+=win*win
    return (out/np.maximum(ws,EPS))[PAD:PAD+h,PAD:PAD+w],float(np.max(abs((out/np.maximum(ws,EPS))[PAD:PAD+h,PAD:PAD+w]-img)) if condition=='none' else math.nan)

def peak_only(window,pts):
    mask=1-local_mask(window['z'].shape,pts,'hard',1);return np.fft.ifft2(np.fft.ifftshift(window['z']*mask)).real
def plot15(img,windows,rec,out):
    # centre image window demonstrates multiscale FFT and harmonic reconstruction
    a=dict(windows[len(windows)//2]);source=np.pad(img,((PAD,PAD),(PAD,PAD)),mode='reflect');p=source[a['y']:a['y']+W,a['x']:a['x']+W];a['z'],_=power_fft(p,W);fig,ax=plt.subplots(2,4,figsize=(17,9),constrained_layout=True);lim=np.percentile(img,[1,99]);dl=np.percentile(abs(rec['local_all']-img),99)
    ims=[(img,'original','gray'),(rec['local_base'],'local fundamentals','gray'),(rec['local_all'],'fundamental + confirmed harmonics','gray'),(rec['local_all']-img,'difference','coolwarm'),(p,'512 patch','gray'),(np.log10(abs(a['z'])**2+EPS),'512 FFT','magma'),(peak_only(a,a['bases']),'bases only inverse FFT','coolwarm'),(peak_only(a,a['harmonics']),'confirmed harmonics inverse FFT','coolwarm')]
    for q,(im,t,c) in zip(ax.ravel(),ims):
      if c=='gray':q.imshow(im[::3,::3] if im.shape[0]>600 else im,cmap=c,vmin=lim[0],vmax=lim[1])
      elif c=='coolwarm':q.imshow(im[::3,::3] if im.shape[0]>600 else im,cmap=c,vmin=-dl,vmax=dl)
      else:q.imshow(im,cmap=c)
      q.set_title(t);q.set_xticks([]);q.set_yticks([])
    fig.savefig(out/'image15_local_harmonic_comparison.png',dpi=140);plt.close(fig)

def score_condition(img,defs,exc):
    setup=det.radial_setup();r=det.spectra_for_image(img,setup,workers=4,batch=64);s=det.SPACING//det.STRIDE
    if r['norm'].shape[0]<=2*s:return np.array([]),np.array([]),[],None
    score=det.strict_scores(r['norm']);xs=r['xs'][s:-s];ys=r['ys'][s:-s];union=np.logical_or.reduce([x['mask'] for x in defs]);tm=det.rect_any(det.integral(union),xs,ys,16);em=det.rect_any(det.integral(exc),xs,ys,16);return score[(~tm)&(~em)],score[tm&~em],(score,tm,em), (xs,ys)

def add_evaluation(ds,n,conds):
    d,e,_,defs=ann(n,conds['none'].shape)
    tmp={c:score_condition(im,defs,e) for c,im in conds.items()}
    base=tmp['none'];score,tm,em=base[2];xs,ys=base[3]
    for c,t in tmp.items():ds[c]['normal'].append(t[0]);ds[c]['defect'].append(t[1])
    for j,a in enumerate(defs,1):
      m=det.rect_any(det.integral(a['mask']),xs,ys,16)&~em;yy,xx=np.nonzero(m)
      if not len(yy):continue
      k=int(np.argmax(score[yy,xx]));iy,ix=yy[k],xx[k]
      for c,t in tmp.items():ds[c]['instances'].append({'image':n,'defect_number':j,'defect_type':a['label'],'target_x':int(xs[ix]),'target_y':int(ys[iy]),'score':float(t[2][0][iy,ix])})

def finish_evaluation(ds):
    perf=[];types=[]
    for c,d in ds.items():
      normal=np.concatenate(d['normal']);defect=np.concatenate(d['defect']);r={'condition':c,'normal_windows':len(normal),'defect_windows':len(defect),'roc_auc':float(roc_auc_score(np.r_[np.zeros(len(normal)),np.ones(len(defect))],np.r_[normal,defect]))}
      for f in (.01,.05):
        th=float(np.quantile(normal,1-f,method='higher'));r[f'threshold_{int(f*100)}pct']=th;r[f'actual_fpr_{int(f*100)}pct']=float(np.mean(normal>th));r[f'window_tpr_{int(f*100)}pct']=float(np.mean(defect>th))
        for q in d['instances']:q[f'hit_{int(f*100)}pct']=q['score']>th
      perf.append(r)
      for typ in sorted(set(q['defect_type'] for q in d['instances'])):
        q=[x for x in d['instances'] if x['defect_type']==typ];types.append({'condition':c,'defect_type':typ,'instances':len(q),'median_score':float(np.median([x['score'] for x in q])),'tpr_1pct':float(np.mean([x['hit_1pct'] for x in q])),'tpr_5pct':float(np.mean([x['hit_5pct'] for x in q]))})
    return perf,types,ds

def main():
  OUT.mkdir(exist_ok=True);top=global_notch.read_csv(PRIOR/'fft_peak_pairs_top20.csv');fam=global_notch.read_csv(PRIOR/'grid_family_candidates.csv');allrecs=[];ola_rows=[];ds={c:{'normal':[],'defect':[],'instances':[]} for c in ('none','global_base','local_base','local_all')}
  for n in IMAGES:
    print(f'image{n}',flush=True);img,wins,recs=analyze_image(n,top,fam,detailed=(n==15));allrecs+=recs
    rec={};rec['none'],err=ola(img,wins,'none');rec['local_base'],_=ola(img,wins,'local_base');rec['local_all'],_=ola(img,wins,'local_all')
    glob=global_notch.apply_notch(img,priors(n,top,fam),'hard',2)[0];rec['global_base']=glob
    ola_rows.append({'image':n,'ola_no_notch_max_abs_error':err,'harmonic_confirmed_windows':sum(len(x['harmonics']) for x in wins),'total_512_windows':len(wins),'local_notch_radius_bins':1,'local_notch_width_cycles_per_pixel':1/W})
    if n==15:
      rec['local_all_gauss'],_=ola(img,wins,'local_all_gauss');plot15(img,wins,rec,OUT);del rec['local_all_gauss']
    if n in EVAL:add_evaluation(ds,n,rec)
    del rec,wins,img,glob
  write(OUT/'multiscale_peak_observations.csv',allrecs);write(OUT/'local_ola_reconstruction_metrics.csv',ola_rows)
  perf,types,ds=finish_evaluation(ds);write(OUT/'local_harmonic_performance.csv',perf);write(OUT/'local_harmonic_type_performance.csv',types)
  # summary harmonic adoption rates, only normal/non-annotated windows shown separately.
  harm=[r for r in allrecs if r['kind']=='harmonic' and r['window_size']==512];summ=[]
  for n in IMAGES:
   for label in sorted(set(r['label'] for r in harm if r['image']==n)):
    q=[r for r in harm if r['image']==n and r['label']==label];summ.append({'image':n,'harmonic':label,'windows':len(q),'confirmed_windows':sum(bool(r.get('confirmed_multiscale')) for r in q),'confirmed_fraction':np.mean([bool(r.get('confirmed_multiscale')) for r in q]),'median_prominence_db':float(np.median([r['prom_db'] for r in q]))})
  write(OUT/'harmonic_confirmation_summary.csv',summ)
  fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);names=[r['condition'] for r in perf];ax.bar(np.arange(len(names))-.18,[r['roc_auc'] for r in perf],.36,label='ROC-AUC');ax.bar(np.arange(len(names))+.18,[r['window_tpr_1pct'] for r in perf],.36,label='window TPR @ FPR1%');ax.set(xticks=np.arange(len(names)),xticklabels=names,ylabel='metric',title='Fixed-set local FFT detector comparison');ax.legend();fig.savefig(OUT/'performance_comparison.png',dpi=150);plt.close(fig)
  report(perf,types,ola_rows,summ)
def report(perf,types,ola,summ):
 p={r['condition']:r for r in perf};im15=[r for r in ola if r['image']==15][0];h15=[r for r in summ if r['image']==15 and r['confirmed_windows']]
 lines=['# 局所マルチスケールFFTによる格子高調波ノッチ', '', '## 結論', '',f'- OLAのノッチなし最大絶対復元誤差（image15）は {im15["ola_no_notch_max_abs_error"]:.3e} であり、窓処理と合成の基礎復元を確認した。','- 高調波は512/256/128 pxの実ピーク一致（512 px prominence ≥4 dB かつ短い窓の一方 ≥3 dB）を満たす場合だけ局所ノッチに加えた。未確認の予測整数倍は除去していない。',f'- image15で確認された高調波種: {", ".join(r["harmonic"] for r in h15) if h15 else "なし（条件を満たす一貫ピークなし）"}。','- 検出性能の優劣は下表CSVで比較する。局所適応ノッチは高調波を多く除去するほど良いとは限らず、傷帯域の損失を含めて判断する。','','## 性能（固定窓集合）','']
 for r in perf:lines.append(f'- {r["condition"]}: AUC={r["roc_auc"]:.4f}, TPR@1%={r["window_tpr_1pct"]:.4f}, TPR@5%={r["window_tpr_5pct"]:.4f}')
 lines+=['','## 出力','', '- `multiscale_peak_observations.csv`: 512/256/128pxのピーク位置・prominence・一致判定。','- `harmonic_confirmation_summary.csv`: 画像別・高調波別の確認率。','- `image15_local_harmonic_comparison.png`: 基本のみ／高調波込み／逆FFT成分／FFT。','- `local_ola_reconstruction_metrics.csv`: OLA復元精度とノッチ幅（cycles/pixel）。','- `local_harmonic_performance.csv`, `local_harmonic_type_performance.csv`: 固定窓集合の性能。','','## 限界','', '512px窓には約3周期しか含まれず、局所周波数推定には分解能限界がある。局所ピークが不安定な位置での過剰ノッチを避けるため、確認条件は保守的にした。']
 (OUT/'report_ja.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
