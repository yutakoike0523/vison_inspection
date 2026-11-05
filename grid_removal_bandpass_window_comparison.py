#!/usr/bin/env python3
"""Grid-frequency removal followed by fixed 128px band-pass reconstructions.

Analysis-only script.  It writes a new directory and never changes earlier
experiments.  Peak localisation uses mean removal + 2-D Hann.  The notch is
applied to the non-windowed complex FFT of the same mean-removed image.
"""
from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

import analysis_fft_normalization_128 as det

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data20260903'/'data20260903'
OUT=ROOT/'analysis_grid_removal_bandpass_002_005'
EPS=1e-20; N=128; HALF=64; BAND=(.02,.05)
FONT={'font.family':['Yu Gothic','Meiryo','DejaVu Sans']}

def csvwrite(path, rows, fields=None):
    cols=list(dict.fromkeys(k for r in rows for k in r)) if rows else (fields or ['status'])
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def load(n): return np.asarray(Image.open(DATA/f'image{n}.png').convert('L'),float)
def hann2(shape): return np.outer(np.hanning(shape[0]),np.hanning(shape[1]))
def conj(y,x,shape): return ((-y)%shape[0],(-x)%shape[1])
def fcoord(y,x,shape):
    h,w=shape; return ((x-w//2)/w,(y-h//2)/h)
def radius(a,b): return math.hypot(a,b)
def sep(a,b):
    # orientation distance: g and -g represent the same grid direction.
    d=abs((math.degrees(math.atan2(a[1],a[0])-math.atan2(b[1],b[0]))+90)%180-90)
    return d

def peaks(gray, limit=80):
    """Canonical, conjugate-paired local maxima in a Hann spectrum."""
    h,w=gray.shape; x=(gray-gray.mean())*hann2(gray.shape)
    z=np.fft.fftshift(np.fft.fft2(x)); p=np.abs(z)**2; db=10*np.log10(p+EPS)
    cy,cx=h//2,w//2; yy,xx=np.ogrid[:h,:w]; rr=np.hypot(yy-cy,xx-cx)
    loc=(p==maximum_filter(p,size=5,mode='nearest')) & (rr>=3)
    # Keep an intentionally broad list; later rules decide what is grid-like.
    ids=np.argwhere(loc); ids=ids[np.argsort(p[loc])[::-1]]
    got=[]; seen=set()
    for y,x0 in ids:
        sy,sx=conj(int(y),int(x0),gray.shape); key=tuple(sorted(((int(y),int(x0)),(sy,sx))))
        if key in seen: continue
        seen.add(key)
        fx,fy=fcoord(int(y),int(x0),gray.shape); fr=radius(fx,fy)
        if fr==0: continue
        # 3--6 bin annulus prominence, as specified.
        d=np.hypot(yy-y,xx-x0); ring=(d>=3)&(d<=6)
        prom=float(db[y,x0]-np.median(db[ring]))
        got.append(dict(y=int(y),x=int(x0),sym_y=sy,sym_x=sx,fx=fx,fy=fy,
                        frequency=fr,period_px=1/fr,angle_deg=math.degrees(math.atan2(fy,fx)),
                        power=float(p[y,x0]),power_db=float(db[y,x0]),prominence_db=prom))
        if len(got)>=limit: break
    for i,r in enumerate(got,1): r['peak_id']=i
    return got,z,p

def select_base(rows):
    usable=[r for r in rows if 25<=r['period_px']<=250 and r['prominence_db']>=4]
    candidates=[]
    for a,b in itertools.combinations(usable,2):
        ang=sep((a['fx'],a['fy']),(b['fx'],b['fy']))
        if 60<=ang<=120:
            candidates.append((a['power']+b['power'],a,b,ang))
    if not candidates: return [],None
    _,a,b,ang=max(candidates,key=lambda q:q[0]); return [a,b],ang

def predicted(base, shape):
    # Unique conjugate representatives, lowest order kept in duplicate cases.
    h,w=shape; out={}
    for m in range(-8,9):
      for n in range(-8,9):
        order=abs(m)+abs(n)
        if not (2<=order<=8): continue
        fx=m*base[0]['fx']+n*base[1]['fx']; fy=m*base[0]['fy']+n*base[1]['fy']; fr=radius(fx,fy)
        if not (.004375<=fr<=.065625) or abs(fx)>.5 or abs(fy)>.5: continue
        # reduce aliases in the sampling interval and choose the canonical conjugate pair.
        ix=int(round(fx*w))%w; iy=int(round(fy*h))%h; sy,sx=conj(iy,ix,shape)
        key=tuple(sorted(((iy,ix),(sy,sx))))
        typ='harmonic' if (m==0) != (n==0) else 'mixed'
        label=(f'{abs(m)}g1' if n==0 else f'{abs(n)}g2' if m==0 else f'{m}g1{n:+d}g2')
        item=dict(m=m,n=n,order=order,classification=typ,label=label,pred_y=iy,pred_x=ix,pred_fx=fx,pred_fy=fy,frequency=fr)
        if key not in out or order<out[key]['order']: out[key]=item
    return list(out.values())

def match_components(pred, observed, shape):
    # Global one-to-one greedy by bin distance then lower lattice order.
    proposals=[]
    for j,q in enumerate(pred):
      for i,p in enumerate(observed):
        d=min(math.hypot(q['pred_y']-p['y'],q['pred_x']-p['x']),math.hypot(q['pred_y']-p['sym_y'],q['pred_x']-p['sym_x']))
        if d<=2 and p['prominence_db']>=4: proposals.append((d,q['order'],-p['power'],j,i))
    usedq=set(); usedp=set(); ans=[]
    for d,_,_,j,i in sorted(proposals):
      if j in usedq or i in usedp: continue
      usedq.add(j); usedp.add(i); ans.append((pred[j],observed[i],d))
    return ans

def local_confirm(gray, ann, q, base_shape, win):
    """Label-audited local repetition, without changing selection after audit."""
    h,w=gray.shape; bad=np.zeros((h,w),bool)
    for a in ann:
        if a['label'] in det.DEFECT_LABELS+det.EXCLUSION_LABELS: bad|=a['mask']
    count=hits=0; yy,xx=np.ogrid[:win,:win]
    for y0 in range(0,h-win+1,128):
      for x0 in range(0,w-win+1,128):
        if bad[y0:y0+win,x0:x0+win].any(): continue
        count+=1; patch=gray[y0:y0+win,x0:x0+win]; z=np.fft.fftshift(np.fft.fft2((patch-patch.mean())*hann2((win,win))))
        p=np.abs(z)**2; db=10*np.log10(p+EPS)
        py=int(round(q['pred_fy']*win))+win//2; px=int(round(q['pred_fx']*win))+win//2
        if not (0<=py<win and 0<=px<win): continue
        d=np.hypot(yy-py,xx-px); ring=(d>=3)&(d<=6); near=d<=2
        prom=10*np.log10((p[near].max()+EPS)/(np.median(p[ring])+EPS))
        if prom>=4: hits+=1
    return count,hits,(hits/count if count else math.nan)

def notch_mask(shape, components):
    h,w=shape; yy,xx=np.ogrid[:h,:w]; m=np.ones(shape,float)
    for p in components:
      for y,x in ((p['y'],p['x']),(p['sym_y'],p['sym_x'])):
        m[(yy-y)**2+(xx-x)**2<=4]=0
    return m

def filter_image(gray, components):
    mean=gray.mean(); z=np.fft.fftshift(np.fft.fft2(gray-mean)); m=notch_mask(gray.shape,components)
    removed=np.fft.ifft2(np.fft.ifftshift(z*(1-m))).real
    kept=np.fft.ifft2(np.fft.ifftshift(z*m)).real+mean
    return kept,removed,m,z

def center_of(mask):
    y,x=np.nonzero(mask); return (int(round(x.mean())),int(round(y.mean())))
def valid_center(x,y,shape): return HALF<=x<=shape[1]-HALF and HALF<=y<=shape[0]-HALF
def window_labels(anns, x,y,shape):
    x0,x1=x-HALF,x+HALF; y0,y1=y-HALF,y+HALF
    if x0<0 or y0<0 or x1>shape[1] or y1>shape[0]: return 'out_of_image',''
    labels=[a['label'] for a in anns if a['mask'][y0:y1,x0:x1].any()]
    if any(t in det.DEFECT_LABELS for t in labels): return 'defect','|'.join(labels)
    if any(t in det.EXCLUSION_LABELS for t in labels): return 'tag/background','|'.join(labels)
    return 'clean',''

def bandpass(patch):
    x=patch-patch.mean(); x*=hann2((N,N)); z=np.fft.fftshift(np.fft.fft2(x)); f=np.fft.fftshift(np.fft.fftfreq(N)); fy,fx=np.meshgrid(f,f,indexing='ij'); r=np.hypot(fx,fy)
    b=np.fft.ifft2(np.fft.ifftshift(np.where((r>=BAND[0])&(r<=BAND[1]),z,0))).real
    return b,float(np.sqrt(np.mean(b*b)))

def save_full(dirp, original, imgs, rem, mask, z, components):
    lim=np.percentile(original,[1,99]); dl=max(np.percentile(np.abs(v),99.5) for v in rem.values())
    fig,ax=plt.subplots(2,4,figsize=(18,9),constrained_layout=True)
    entries=[(original,'Original',False)]+[(imgs[k],k,False) for k in ('A_none','B_base','C_harmonics','D_all')]+[(rem[k],f'Removed: {k}',True) for k in ('B_base','C_harmonics','D_all')]
    for a,(im,title,diff) in zip(ax.ravel(),entries):
      a.imshow(im,cmap='coolwarm' if diff else 'gray',vmin=-dl if diff else lim[0],vmax=dl if diff else lim[1]); a.set_title(title);a.axis('off')
    fig.savefig(dirp/'full_image_comparison.png',dpi=150);plt.close(fig)
    for key,name in [('B_base','base'),('C_harmonics','harmonics'),('D_all','all')]:
      Image.fromarray(np.clip(imgs[key],0,255).astype(np.uint8)).save(dirp/f'filtered_{name}.png')
      np.save(dirp/f'filtered_{name}.npy',imgs[key]); np.save(dirp/f'removed_components_{name}.npy',rem[key])
      fig,ax=plt.subplots(figsize=(8,6)); ax.imshow(rem[key],cmap='coolwarm',vmin=-dl,vmax=dl);ax.axis('off');ax.set_title(f'Removed components: {name}');fig.savefig(dirp/f'removed_components_{name}.png',dpi=150);plt.close(fig)
    Image.fromarray(original.astype(np.uint8)).save(dirp/'original.png')
    fig,ax=plt.subplots(1,2,figsize=(12,5),constrained_layout=True); db=10*np.log10(np.abs(z)**2+EPS);v=np.percentile(db,[5,99.8]); ax[0].imshow(db,cmap='magma',vmin=v[0],vmax=v[1]); ax[1].imshow(mask,cmap='gray',vmin=0,vmax=1)
    for p in components:
      for y,x in ((p['y'],p['x']),(p['sym_y'],p['sym_x'])): ax[0].plot(x,y,'co',ms=4);ax[1].plot(x,y,'r.',ms=3)
    for a,t in zip(ax,['Selected FFT peaks','Hard-notch mask (white=retained)']): a.set_title(t);a.set_xlim(z.shape[1]//2-100,z.shape[1]//2+100);a.set_ylim(z.shape[0]//2+100,z.shape[0]//2-100)
    fig.savefig(dirp/'fft_notch_mask.png',dpi=180);plt.close(fig)

def save_windows(dirp, image_no, original, imgs, anns, target):
    x,y=target; centers=[(x,y)]+[(x+dx*128,y+dy*128) for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy]
    names=['D']+[f'R{i}' for i in range(1,9)]; cond=['A_none','B_base','C_harmonics','D_all']; data={}; rows=[]
    for name,(cx,cy) in zip(names,centers):
      cls,over=window_labels(anns,cx,cy,original.shape)
      for c in cond:
        if valid_center(cx,cy,original.shape): b,r=bandpass(imgs[c][cy-HALF:cy+HALF,cx-HALF:cx+HALF])
        else: b=np.full((N,N),np.nan);r=math.nan
        data[(name,c)]=(b,r,cls,over,cx,cy)
        rows.append(dict(image=image_no,window=name,condition=c,center_x=cx,center_y=cy,classification=cls,overlap_labels=over,band_rms=r))
    csvwrite(dirp/'window_band_rms.csv',rows)
    valid=[v[0] for v in data.values() if np.isfinite(v[0]).all()]; bl=max(np.percentile(np.abs(a),99.5) for a in valid)
    # Fixed-scale A--D x D/R1..R8 matrix requested in the brief.
    fig,ax=plt.subplots(4,9,figsize=(20,9),constrained_layout=True)
    for i,c in enumerate(cond):
      for j,name in enumerate(names):
        b,r,cls,_,_,_=data[(name,c)]; ax[i,j].imshow(b,cmap='coolwarm',vmin=-bl,vmax=bl);ax[i,j].set_title(f'{c[0]} {name}\n{cls}, {r:.2f}' if np.isfinite(r) else f'{c[0]} {name}\nout');ax[i,j].axis('off')
    fig.savefig(dirp/'bandpass_window_comparison_shared_scale.png',dpi=160);plt.close(fig)
    # Each condition: locations, spatial windows, band images and RMS bars.
    lim=np.percentile(original,[1,99])
    for c in cond:
      fig,ax=plt.subplots(3,9,figsize=(20,7),constrained_layout=True); ax[0,0].imshow(original,cmap='gray',vmin=lim[0],vmax=lim[1]);ax[0,0].set_title('Window locations')
      for name,(cx,cy) in zip(names,centers):
        col={'defect':'red','clean':'lime','tag/background':'orange','out_of_image':'gray'}[data[(name,c)][2]]; ax[0,0].add_patch(plt.Rectangle((cx-HALF,cy-HALF),N,N,fill=False,ec=col,lw=1.5));ax[0,0].text(cx-HALF,cy-HALF,name,color=col,fontsize=7)
      ax[0,0].axis('off')
      for j,name in enumerate(names):
        b,r,cls,_,cx,cy=data[(name,c)];
        if valid_center(cx,cy,original.shape): ax[1,j].imshow(imgs[c][cy-HALF:cy+HALF,cx-HALF:cx+HALF],cmap='gray',vmin=lim[0],vmax=lim[1]);ax[2,j].imshow(b,cmap='coolwarm',vmin=-bl,vmax=bl)
        ax[1,j].set_title(name);ax[1,j].axis('off');ax[2,j].axis('off')
      rms=[data[(n,c)][1] for n in names];ax[0,1].bar(range(9),rms,color=['red']+['lime' if data[(n,c)][2]=='clean' else 'orange' for n in names[1:]]);ax[0,1].set_xticks(range(9),names);ax[0,1].set_title('0.02–0.05 RMS')
      for k in range(2,9):ax[0,k].axis('off')
      fig.savefig(dirp/f'window_comparison_{c[0]}.png',dpi=150);plt.close(fig)
    # comparison summary / crop / metrics
    clean=[n for n in names[1:] if data[(n,'A_none')][2]=='clean']; summary=[]
    for c in cond:
      d=data[('D',c)][1]; cr=np.array([data[(n,c)][1] for n in clean],float); med=float(np.median(cr)) if len(cr) else math.nan
      summary.append(dict(image=image_no,condition=c,defect_rms=d,clean_count=len(clean),clean_rms_median=med,rms_difference=d-med if np.isfinite(med) else math.nan,rms_ratio=d/(med+EPS) if np.isfinite(med) else math.nan))
    csvwrite(dirp/'defect_vs_reference_comparison.csv',summary)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for c in cond: ax.plot(names,[data[(n,c)][1] for n in names],marker='o',label=c)
    for s in summary:
      if np.isfinite(s['clean_rms_median']):ax.axhline(s['clean_rms_median'],ls=':',alpha=.4)
    ax.set(ylabel='band-pass RMS',title='RMS change, same 128px windows');ax.legend();fig.savefig(dirp/'rms_comparison.png',dpi=170);plt.close(fig)
    x0=max(0,x-128);x1=min(original.shape[1],x+128);y0=max(0,y-128);y1=min(original.shape[0],y+128)
    fig,ax=plt.subplots(1,4,figsize=(14,4),constrained_layout=True)
    for a,c in zip(ax,cond):a.imshow(imgs[c][y0:y1,x0:x1],cmap='gray',vmin=lim[0],vmax=lim[1]);a.set_title(c);a.axis('off')
    fig.savefig(dirp/'defect_crop_comparison.png',dpi=170);plt.close(fig)
    return rows,summary

def report(out, selected, hrows, summaries, recs):
    lines=['# 格子除去後の 0.02–0.05 cycles/pixel 局所逆FFT比較','','## 方法','- ピーク探索: 平均除去＋2次元Hann窓。復元ノッチ: 非Hannの平均除去複素FFTにhard notch半径2 bin。','- 基本波は周期25–250 px、突出度4 dB以上、方向差60–120°の組から強度和最大の2組。','- 高調波／混合成分は予測位置2 bin以内・突出度4 dB以上・局所再現性（5窓以上かつ50%以上）を全て満たす場合のみ採用。','- 128px局所処理は平均除去＋Hann、0.02–0.05 cycles/pixelのみを位相保持で逆FFTした。全条件で窓座標と共通発散スケールを固定。','']
    for n in (15,18):
      ss=[r for r in selected if r['image']==n]; hs=[r for r in hrows if r['image']==n and r['adopted']=='yes']; q=[r for r in summaries if r['image']==n]
      lines += [f'## image{n}',f"基本波: " + ('; '.join(f"({r['fx']:.6f}, {r['fy']:.6f}) cyc/px, 周期{r['period_px']:.1f}px" for r in ss) if ss else '条件を満たす2組なし。'),f'採用した高調波・混合成分: {len(hs)} 個。image18では256pxの独立評価窓が5未満なら採用しないルールで判定。']
      for r in q: lines.append(f"- {r['condition']}: 傷RMS={r['defect_rms']:.4g}, clean中央値={r['clean_rms_median']:.4g}, 比={r['rms_ratio']:.4g}")
      lines.append('解釈: RMS比は局所的な補助指標であり、傷検出性能の向上を直接示すものではない。')
    lines += ['','## 検証','`reconstruction_validation.csv` の誤差は「除去成分＋残差−元画像」の最大絶対値。選択FFT座標・ノッチbin数は各画像のCSVに保存した。']
    (out/'report_ja.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    plt.rcParams.update(FONT); OUT.mkdir(exist_ok=True)
    selected=[]; harmonic=[]; notchrows=[]; recon=[]; allwin=[]; compares=[]
    for n,defidx in ((15,0),(18,2)):
      print(f'image{n}',flush=True); gray=load(n); anns=det.load_annotations(DATA/f'image{n}.json',gray.shape); obs,_,_=peaks(gray)
      base,ang=select_base(obs)
      idir=OUT/f'image{n}'; idir.mkdir(exist_ok=True)
      for r in obs:r['image']=n
      csvwrite(idir/'detected_peak_pairs.csv',obs)
      for r in base: selected.append(dict(**r,base_angle_separation_deg=ang))
      pred=predicted(base,gray.shape) if len(base)==2 else []
      matched=match_components(pred,obs,gray.shape)
      matched_ids={id(q) for q,p,d in matched}
      adopted=[]; win=512 if n==15 else 256
      # Preserve rejected predictions too: an empty table would otherwise make
      # "no adopted harmonic" indistinguishable from "no search performed".
      for q in pred:
        if id(q) not in matched_ids:
          harmonic.append(dict(image=n,window_size=win,observed_peak_id='',match_distance_bins='',local_windows='',local_hits='',local_confirmation_rate='',adopted='no',rejection_reason='no_measured_peak_within_2_bins',**q,observed_fx='',observed_fy='',prominence_db='',power=''))
      for q,p,d in matched:
        num,hit,rate=local_confirm(gray,anns,q,gray.shape,win)
        ok=num>=5 and rate>=.5
        row=dict(image=n,window_size=win,observed_peak_id=p['peak_id'],match_distance_bins=d,local_windows=num,local_hits=hit,local_confirmation_rate=rate,adopted='yes' if ok else 'no',rejection_reason='' if ok else 'local_repeatability_not_met',**q,observed_fx=p['fx'],observed_fy=p['fy'],prominence_db=p['prominence_db'],power=p['power'])
        harmonic.append(row)
        if ok: adopted.append(p)
      csvwrite(idir/'harmonic_validation.csv',[r for r in harmonic if r['image']==n])
      groups={'A_none':[],'B_base':base,'C_harmonics':base+[p for q,p,d in matched if q['classification']=='harmonic' and next(r for r in harmonic if r['image']==n and r['observed_peak_id']==p['peak_id'])['adopted']=='yes'], 'D_all':base+adopted}
      # remove duplicate peak pairs after category unions
      for k in groups:
        uniq={p['peak_id']:p for p in groups[k]};groups[k]=list(uniq.values())
      imgs={'A_none':gray.copy()}; rem={'A_none':np.zeros_like(gray)}; masks={}; zs={}
      for c,comps in groups.items():
        if c=='A_none':continue
        imgs[c],rem[c],masks[c],zs[c]=filter_image(gray,comps)
        e=np.max(np.abs(imgs[c]+rem[c]-gray)); notchrows.append(dict(image=n,condition=c,peak_pairs=len(comps),removed_fft_bins=int((masks[c]==0).sum()),notch_radius_bins=2,notch_width_cpp_x=2/gray.shape[1],notch_width_cpp_y=2/gray.shape[0]));recon.append(dict(image=n,condition=c,max_abs_reconstruction_error=e))
      # Make A mask/Z compatible with visualiser.
      mask=masks.get('D_all',np.ones(gray.shape)); z=zs.get('D_all',np.fft.fftshift(np.fft.fft2(gray-gray.mean())))
      save_full(idir,gray,imgs,rem,mask,z,groups['D_all'])
      defects=[a for a in anns if a['label'] in det.DEFECT_LABELS]; target=center_of(defects[min(defidx,len(defects)-1)]['mask'])
      wr,cr=save_windows(idir,n,gray,imgs,anns,target);allwin+=wr;compares+=cr
    csvwrite(OUT/'selected_grid_frequencies.csv',selected);csvwrite(OUT/'harmonic_validation.csv',harmonic);csvwrite(OUT/'notch_filter_summary.csv',notchrows);csvwrite(OUT/'window_band_rms.csv',allwin);csvwrite(OUT/'defect_vs_reference_comparison.csv',compares);csvwrite(OUT/'reconstruction_validation.csv',recon)
    report(OUT,selected,harmonic,compares,recon)

if __name__=='__main__': main()
