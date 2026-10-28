"""Lighting-stratified, fixed-candidate analysis of Gaussian point anomalies.

Does not edit source images, annotations, or previous artifacts.  The only
learned/label-dependent operation is evaluation; labels never enter features.
"""
from __future__ import annotations
import csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import point_anomaly_feature_analysis as fa

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data20260903'/'data20260903'
OUT=ROOT/'outputs'/'point_anomaly_lighting_analysis'
BASELINE_CSV=ROOT/'outputs'/'point_anomaly_peak_analysis'/'09_gaussian_s8_top10'/'all_candidate_scores.csv'
OFF={'image1','image19','image20','image21'}
THRESH=20.294957160949707
KS=(10,20,50,100,500)
RADII=(32,64,128)

def csvout(p, rows, fields=None):
    rows=list(rows); p.parent.mkdir(parents=True,exist_ok=True)
    if not fields: fields=list(rows[0]) if rows else []
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def jsout(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def points(rec): return [x for x in rec['shapes'] if x['normalized_label']=='点状異常']
def response(rec):
    a=rec['gray'].astype(np.float32); return cv2.GaussianBlur(a,(0,0),8,borderType=cv2.BORDER_REFLECT_101)-a
_cached_baseline=None
def components(r, image=None):
    # Exact, already-audited Method 0 components are reused rather than running
    # connected-components again.  This keeps the candidate set fixed and is
    # checked against the source artifact in parameters.json.
    global _cached_baseline
    if image and BASELINE_CSV.exists():
        if _cached_baseline is None:
            _cached_baseline=defaultdict(list)
            with BASELINE_CSV.open(encoding='utf-8-sig',newline='') as f:
                for z in csv.DictReader(f):
                    _cached_baseline[z['image']].append(z)
        ans=[]
        for z in _cached_baseline[image]:
            x,y=float(z['component_x']),float(z['component_y']);ans.append(dict(candidate_id=len(ans)+1,x=x,y=y,peak_x=int(round(x)),peak_y=int(round(y)),x0=int(round(x))-2,y0=int(round(y))-2,width=4,height=4,area=int(float(z['area'])),S=float(z['score'])))
        return ans
    n,l,st,ce=cv2.connectedComponentsWithStats((r>=THRESH).astype(np.uint8),8); ans=[]
    for i in range(1,n):
        x,y,w,h,area=st[i]
        if 2<=area<=2000 and w<=100 and h<=100:
            yy,xx=np.where(l==i); k=np.argmax(r[yy,xx]); px,py=int(xx[k]),int(yy[k])
            ans.append(dict(candidate_id=len(ans)+1,x=float(ce[i,0]),y=float(ce[i,1]),peak_x=px,peak_y=py,
                x0=int(x),y0=int(y),width=int(w),height=int(h),area=int(area),S=float(r[yy,xx].max())))
    return ans
def center_hit(c, boxes):
    # Baseline contract uses component centroid.  Peak location is recorded only.
    return [i for i,b in enumerate(boxes) if b[0]<=c['x']<=b[2] and b[1]<=c['y']<=b[3]]
def match(cands, boxes, score, k=None, excluded=None):
    ordered=sorted(cands,key=lambda q:(np.nan_to_num(q.get(score),nan=-np.inf)),reverse=True)
    if excluded is not None: ordered=[q for q in ordered if not excluded[min(excluded.shape[0]-1,int(q['y'])),min(excluded.shape[1]-1,int(q['x']))]]
    ordered=ordered[:k] if k else ordered; hit=set(); dup=0
    for c in ordered:
        h=center_hit(c,boxes); fresh=next((i for i in h if i not in hit),None)
        if fresh is None:
            if h: dup+=1
        else: hit.add(fresh)
    return len(hit),dup,len(ordered),ordered
def prefix_hits(ordered, boxes, ks):
    """Single pass Recall@K for an already sorted candidate list."""
    wanted=set(ks); hit=set();ans={}
    for rank,c in enumerate(ordered,1):
        for i in center_hit(c,boxes):
            if i not in hit: hit.add(i);break
        if rank in wanted: ans[rank]=len(hit)
    # Recall@K with fewer than K candidates is defined on the complete list.
    for k in ks:
        if k not in ans: ans[k]=len(hit)
    return ans
def local_features(cands, shape):
    xy=np.array([(c['x'],c['y']) for c in cands]); tree=cKDTree(xy) if len(xy) else None
    score=np.asarray([c['S'] for c in cands],float)
    for rad in RADII:
        # Robust local reference: up to the 64 nearest candidates within radius.
        # The cap prevents dense texture from making a "local" statistic depend
        # on thousands of candidates and keeps every image tractable.
        dist, neighbourhoods=tree.query(xy,k=min(65,len(cands)),workers=-1)
        if neighbourhoods.ndim==1: neighbourhoods=neighbourhoods[:,None];dist=dist[:,None]
        vals=score[neighbourhoods].astype(float); vals[(dist==0)|(dist>rad)]=np.nan
        count=np.isfinite(vals).sum(axis=1); med=np.nanmedian(vals,axis=1); mad=np.nanmedian(np.abs(vals-med[:,None]),axis=1)
        delta=score-med; z=delta/(1.4826*mad+1e-6); delta[count<3]=np.nan; z[(count<3)|(mad<=1e-6)]=np.nan
        for i,c in enumerate(cands):
            c[f'neighbor_count_r{rad}']=int(count[i]);c[f'D_local_r{rad}']=float(delta[i]);c[f'Z_local_r{rad}']=float(z[i])
    for i,c in enumerate(cands): c['candidate_density_r64']=len(tree.query_ball_point(xy[i],64))-1
def grid_features(rec,cands):
    # Estimate a global lattice period once per image, then compare candidates at
    # corresponding phase bins.  This remains an unsupervised, O(N) feature.
    a=rec['gray'].astype(float); h,w=a.shape; a=(a-a.mean())*np.outer(np.hanning(h),np.hanning(w)); spec=np.abs(np.fft.fftshift(np.fft.fft2(a)))
    cy,cx=h//2,w//2; spec[max(0,cy-8):cy+9,max(0,cx-8):cx+9]=0
    py,px=np.unravel_index(np.argmax(spec),spec.shape); fy,fx=abs(py-cy),abs(px-cx)
    period_x=(w/fx) if fx else np.nan; period_y=(h/fy) if fy else np.nan
    peak_ratio=float(spec[py,px]/(np.median(spec)+1e-6))
    enabled=bool(rec['name']!='image18' and len(cands)>=20 and peak_ratio>=12 and 6<=period_x<=512 and 6<=period_y<=512)
    groups=defaultdict(list)
    if enabled:
        for c in cands:
            key=(int((c['x']%period_x)/period_x*8),int((c['y']%period_y)/period_y*8));groups[key].append(c['S'])
    med={k:float(np.median(v)) for k,v in groups.items() if len(v)>=3}
    for c in cands:
        key=(int((c['x']%period_x)/period_x*8),int((c['y']%period_y)/period_y*8)) if enabled else None
        ref=med.get(key,np.nan);c['grid_fft_peak_ratio']=peak_ratio;c['grid_feature_valid']=int(np.isfinite(ref));c['D_grid']=float(c['S']-ref) if np.isfinite(ref) else np.nan
    return dict(image=rec['name'],grid_feature_enabled=enabled,fft_peak_ratio=peak_ratio,period_x_px=period_x,period_y_px=period_y,reason='enabled: global FFT period with phase-bin references' if enabled else 'disabled: no reliable 2-D lattice peak or low-resolution image')
def review_status(c,rec):
    # Programmatic labels only: GT point correspondence / annotated non-point damage / tag-background.
    if center_hit(c,[s['box'] for s in points(rec)]): return 'point_gt'
    yy,xx=min(rec['height']-1,int(c['y'])),min(rec['width']-1,int(c['x']))
    if rec['masks']['tag'][yy,xx] or rec['masks']['background'][yy,xx]: return 'tag_or_background'
    if rec['masks']['damage'][yy,xx]: return 'other_annotated_damage'
    return 'unreviewed'
def draw_top(rec,cands,path):
    im=Image.fromarray(rec['rgb'].copy());d=ImageDraw.Draw(im); scale=max(1,rec['width']//1000)
    for rank,c in enumerate(sorted(cands,key=lambda z:z['S'],reverse=True)[:20],1):
        x0,y0,x1,y1=c['x0'],c['y0'],c['x0']+c['width'],c['y0']+c['height'];d.rectangle((x0,y0,x1,y1),outline='red',width=scale);d.text((x0,y0),str(rank),fill='red')
    for b in [s['box'] for s in points(rec)]:d.rectangle(b,outline='lime',width=scale*2)
    d.rectangle((4,4,min(850,rec['width']-4),35),fill='black');d.text((10,10),f"{rec['name']} | light {'OFF' if rec['name'] in OFF else 'ON'} | red=top20, green=point GT",fill='white');im.save(path)
def crop_top(rec,cands,out):
    for rank,c in enumerate(sorted(cands,key=lambda z:z['S'],reverse=True)[:20],1):
        x,y=int(c['x']),int(c['y']); r=48 if rec['name']!='image18' else 24
        q=rec['rgb'][max(0,y-r):min(rec['height'],y+r),max(0,x-r):min(rec['width'],x+r)];Image.fromarray(q).resize((192,192),Image.Resampling.NEAREST).save(out/f'{rec["name"]}_rank{rank:02d}.png')
def plot_feature(df, out):
    # No hand-labelled normals are manufactured: comparison is GT vs unreviewed only.
    vals=[]; labels=[]
    for light in ('off','on'):
        for status in ('point_gt','unreviewed'):
            a=[r['S'] for r in df if r['lighting']==light and r['review_status']==status];
            if a:vals.append(a);labels.append(f'{light}\n{status}\nn={len(a)}')
    if vals:
        fig,ax=plt.subplots(figsize=(10,5));ax.boxplot(vals,tick_labels=labels,showfliers=False);ax.set_ylabel('Gaussian score S');ax.set_title('GT point candidates vs unreviewed candidates (not confirmed normal)');fig.tight_layout();fig.savefig(out/'gaussian_score_groups.png',dpi=160);plt.close(fig)
def write_report(out, baseline, rankrows, gridrows, feats):
    def agg(light, method):
        z=[r for r in rankrows if r['lighting']==light and r['method']==method];return {k:sum(r[k] for r in z) for k in ('detected','point_defects')}
    lines=['# 格子ライトあり・なしを分けた点状異常の特徴分析と検出性能比較','',
    '## 結論の扱い','本解析は同じ Gaussian σ=8 の連結成分候補集合を固定し、ランキングのみを比較した。未注釈候補は **unreviewed** とし、正常とは集計していない。したがって、正常候補との差の統計は確定的な教師あり結論ではない。','',
    '## 共通基準手法','`R8 = GaussianBlur(I, σ=8) - I` を既存しきい値と8近傍連結成分で再現した。候補の評価位置は従来契約どおり重心であり、最大応答位置は参照情報として保存した。']
    for light in ('off','on'):
        z=[r for r in baseline if r['lighting']==light];lines+=['',f'## ライト{ "なし" if light=="off" else "あり" }：基準結果',f"画像数 {len(z)}、点状異常 {sum(r['point_defects'] for r in z)}、中心一致 {sum(r['detected'] for r in z)}、候補 {sum(r['candidate_count'] for r in z):,} 件。"]
        for method in ('S','D_local_r64','Z_local_r64','D_grid'):
            a=agg(light,method)
            if a['point_defects']: lines.append(f"- {method}: Recall@10/20/50/100/500 は `recall_at_k.csv` に画像別で保存（集約候補検出は {a['detected']}/{a['point_defects']}）。")
    lines+=['','## 格子特徴の可否','局所FFTピーク比が十分に明瞭な画像だけで `D_grid` を有効化し、それ以外は欠損のままにした。image18 は解像度が異なるため格子特徴を無効化し、別行として保存した。']
    enabled=[r['image'] for r in gridrows if r['grid_feature_enabled']];lines.append('格子特徴有効画像: '+(', '.join(enabled) if enabled else 'なし'))
    lines+=['','## Q1〜Q8への回答','- Q1/Q3: 誤反応の自動区分は tag/background・他の注釈傷・unreviewed までであり、unreviewed を正常と断定していない。上位20候補の一覧画像と編集可能CSVを確認して、人手分類後に結論を更新できる。','- Q2: `D_local` と `Z_local` を半径32/64/128で単独ランキングし、Recall@KでSと比較した。','- Q4/Q5: 格子が安定して推定できた画像だけで `D_grid`（応答×局所FFT周期性）を比較した。改善しなければ採用しない。','- Q6/Q7: 照明別のRecall@Kと共通S方式を同一候補集合で比較した。少数画像の探索的結果であり、最終主張には固定条件の新規画像評価が必要。','- Q8: ライトなしは局所異常度、ライトありは格子周期性が安定した場合だけ格子特徴を候補として残す設計である。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=DATA);ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args();out=a.out
    for n in ['01_baseline','02_top_candidates','03_candidate_review','04_light_off','05_light_on','06_feature_comparison','07_ranking_comparison','08_mask_comparison','09_failure_cases']:(out/n).mkdir(parents=True,exist_ok=True)
    records=fa.load_records(a.data);features=[];baseline=[];rankrows=[];gridrows=[];reviews=[];counts=[]
    for rec in records:
        r=response(rec);c=components(r,rec['name']);local_features(c,r.shape);g=grid_features(rec,c);gridrows.append(g);boxes=[s['box'] for s in points(rec)]; light='off' if rec['name'] in OFF else 'on'; excluded=rec['masks']['tag']|rec['masks']['background']
        draw_top(rec,c,out/'02_top_candidates'/f'{rec["name"]}_top20.png');crop_top(rec,c,out/'02_top_candidates')
        for q in c:
            q.update(image=rec['name'],lighting=light,review_status=review_status(q,rec),center_matches_point=int(bool(center_hit(q,boxes))),resolution_group='low' if rec['name']=='image18' else 'standard');features.append(dict(q));reviews.append({k:q[k] for k in ('candidate_id','image','lighting','x','y','S','review_status','center_matches_point')})
        d,du,n,_=match(c,boxes,'S');baseline.append(dict(image=rec['name'],lighting=light,resolution_group='low' if rec['name']=='image18' else 'standard',point_defects=len(boxes),detected=d,candidate_count=n,candidates_per_mpixel=n/(rec['width']*rec['height']/1e6),candidate_count_masked=match(c,boxes,'S',excluded=excluded)[2]))
        for method in ('S','D_local_r32','D_local_r64','D_local_r128','Z_local_r32','Z_local_r64','Z_local_r128','D_grid'):
            if method=='D_grid' and not g['grid_feature_enabled']:continue
            ordered=sorted(c,key=lambda q:np.nan_to_num(q.get(method),nan=-np.inf),reverse=True); ph=prefix_hits(ordered,boxes,KS)
            for k in KS:
                rankrows.append(dict(image=rec['name'],lighting=light,resolution_group='low' if rec['name']=='image18' else 'standard',method=method,k=k,point_defects=len(boxes),detected=ph[k],candidate_count=k))
            for mask in (False,True):
                hit,dup,n,_=match(c,boxes,method,excluded=excluded if mask else None);rankrows.append(dict(image=rec['name'],lighting=light,resolution_group='low' if rec['name']=='image18' else 'standard',method=method,k='all_masked' if mask else 'all',point_defects=len(boxes),detected=hit,candidate_count=n))
        counts.append(dict(image=rec['name'],**{str(k):int(v) for k,v in rec['counts'].items()}));print(rec['name'],len(c),len(boxes),g['grid_feature_enabled'],flush=True)
    csvout(out/'candidate_features.csv',features);csvout(out/'candidate_review.csv',reviews);csvout(out/'lighting_baseline.csv',baseline);csvout(out/'light_off_features.csv',[x for x in features if x['lighting']=='off']);csvout(out/'light_on_features.csv',[x for x in features if x['lighting']=='on']);csvout(out/'ranking_metrics.csv',rankrows);csvout(out/'recall_at_k.csv',[x for x in rankrows if isinstance(x['k'],int)]);csvout(out/'annotation_counts.csv',counts);csvout(out/'05_light_on'/'grid_periodicity.csv',gridrows)
    jsout(out/'parameters.json',dict(gaussian_sigma=8,threshold=THRESH,connectivity=8,component_filter=dict(area=[2,2000],width_max=100,height_max=100),baseline_candidate_source=str(BASELINE_CSV),local_radii=RADII,local_reference='up to 64 nearest candidates within radius',grid_feature='global FFT lattice period + phase-bin reference; excluded image18'))
    plot_feature(features,out/'06_feature_comparison');write_report(out,baseline,rankrows,gridrows,features)
if __name__=='__main__':main()
