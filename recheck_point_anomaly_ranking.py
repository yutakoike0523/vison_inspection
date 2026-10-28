"""Corrected-centroid recheck of existing Method-0 ranking features.

No detector, threshold, filter, or feature parameter is tuned here.
"""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import audit_point_anomaly_evaluation as audit
import point_anomaly_feature_analysis as fa

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data20260903'/'data20260903'; OUT=ROOT/'outputs'/'point_anomaly_ranking_recheck'
OFF={'image1','image19','image20','image21'}; PILOT={'image16','image18','image20'}; KS=(10,20,50,100,500); RADII=(32,64,128)
OLD=ROOT/'outputs'/'point_anomaly_peak_analysis'/'09_gaussian_s8_top10'/'all_candidate_scores.csv'
def write(p,rows):
 rows=list(rows);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
def inside(x,y,b):return b[0]<=x<=b[2] and b[1]<=y<=b[3]
def positions(c):return np.array([(x['cx'],x['cy']) for x in c],float)
def local(c):
 xy=positions(c);tree=__import__('scipy').spatial.cKDTree(xy);score=np.array([x['score'] for x in c],float);dist,idx=tree.query(xy,k=min(65,len(c)),workers=-1)
 if idx.ndim==1:idx=idx[:,None];dist=dist[:,None]
 for rad in RADII:
  v=score[idx].astype(float);v[(dist==0)|(dist>rad)]=np.nan;cnt=np.isfinite(v).sum(1);med=np.nanmedian(v,1);mad=np.nanmedian(abs(v-med[:,None]),1);d=score-med;z=d/(1.4826*mad+1e-6);d[cnt<3]=np.nan;z[(cnt<3)|(mad<=1e-6)]=np.nan
  for i,q in enumerate(c):q[f'D_local_r{rad}']=float(d[i]);q[f'Z_local_r{rad}']=float(z[i]);q[f'neighbor_count_r{rad}']=int(cnt[i])
def grid(rec,c):
 a=rec['gray'].astype(float);h,w=a.shape;a=(a-a.mean())*np.outer(np.hanning(h),np.hanning(w));sp=np.abs(np.fft.fftshift(np.fft.fft2(a)));cy,cx=h//2,w//2;sp[max(0,cy-8):cy+9,max(0,cx-8):cx+9]=0;py,px=np.unravel_index(np.argmax(sp),sp.shape);fy,fx=abs(py-cy),abs(px-cx);xp=w/fx if fx else np.nan;yp=h/fy if fy else np.nan;ratio=float(sp[py,px]/(np.median(sp)+1e-6));ok=bool(rec['name']!='image18' and len(c)>=20 and ratio>=12 and 6<=xp<=512 and 6<=yp<=512)
 groups=defaultdict(list)
 if ok:
  for q in c:groups[(int((q['cx']%xp)/xp*8),int((q['cy']%yp)/yp*8))].append(q['score'])
 med={k:float(np.median(v)) for k,v in groups.items() if len(v)>=3}
 for q in c:
  k=(int((q['cx']%xp)/xp*8),int((q['cy']%yp)/yp*8)) if ok else None;v=med.get(k,np.nan);q['D_grid']=float(q['score']-v) if np.isfinite(v) else np.nan;q['grid_valid']=int(np.isfinite(v))
 return dict(image=rec['name'],grid_valid_image=int(ok),fft_peak_ratio=ratio,period_x_px=xp,period_y_px=yp,definition='global FFT dominant period; 8x8 phase-bin median S; D_grid=S-median(S_bin)')
def ranks(c,boxes,method):
 def value(q): return q['score'] if method=='S' else q.get(method)
 ordered=sorted(c,key=lambda q:(-np.nan_to_num(value(q),nan=-np.inf),q['cy'],q['cx']));ans={i:None for i in range(len(boxes))};seen=set()
 for rank,q in enumerate(ordered,1):
  for i,b in enumerate(boxes):
   if i not in seen and inside(q['cx'],q['cy'],b):ans[i]=(rank,q);seen.add(i);break
 return ans,ordered
def metric(c,boxes,method,k):
 r,_=ranks(c,boxes,method);return sum(v is not None and v[0]<=k for v in r.values())
def render(rec,ordered,boxes,path,title):
 im=Image.fromarray(rec['rgb'].copy());d=ImageDraw.Draw(im);line=max(2,rec['width']//700)
 for n,q in enumerate(ordered[:10],1):
  x,y=q['cx'],q['cy'];d.ellipse((x-4,y-4,x+4,y+4),outline='red',width=line);d.text((x+5,y+3),str(n),fill='red')
 for b in boxes:d.rectangle(b,outline='lime',width=line)
 d.rectangle((4,4,min(rec['width']-4,850),35),fill='black');d.text((10,10),title,fill='white');im.save(path)
def main():
 for n in ['01_reproduction','02_light_off','03_light_on','04_grid_valid','05_rank_changes','06_top_candidates','07_centroid_comparison','08_failure_cases']:(OUT/n).mkdir(parents=True,exist_ok=True)
 consistency=[];metrics=[];changes=[];summary=[];gridrows=[];coordinate=[]
 old=defaultdict(list)
 with OLD.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):old[r['image']].append(r)
 for rec in fa.load_records(DATA):
  c=audit.cand(rec);boxes=[s['box'] for s in audit.pts(rec)];light='off' if rec['name'] in OFF else 'on';gridrows.append(grid(rec,c));local(c)
  oldkey={(int(float(x['component_x'])),int(float(x['component_y'])),round(float(x['score']),6),int(float(x['area']))) for x in old[rec['name']]};newkey={(q['x'],q['y'],round(q['score'],6),q['area']) for q in c};consistency.append(dict(image=rec['name'],old_candidates=len(oldkey),recomputed_candidates=len(newkey),shared=len(oldkey&newkey),old_only=len(oldkey-newkey),new_only=len(newkey-oldkey)))
  methods=['S','D_local_r32','D_local_r64','D_local_r128','Z_local_r32','Z_local_r64','Z_local_r128']+(['D_grid'] if gridrows[-1]['grid_valid_image'] else [])
  cache={}
  for m in methods:
   rr,ordered=ranks(c,boxes,m);cache[m]=(rr,ordered)
   for k in (*KS,'all'):
    hit=sum(v is not None and (k=='all' or v[0]<=k) for v in rr.values());metrics.append(dict(image=rec['name'],lighting=light,split='pilot' if rec['name'] in PILOT else 'evaluation',resolution_group='low' if rec['name']=='image18' else 'standard',feature=m,k=k,detected=hit,point_defects=len(boxes),candidate_count=len(c)))
  for i,b in enumerate(boxes,1):
   row=dict(image=rec['name'],lighting=light,split='pilot' if rec['name'] in PILOT else 'evaluation',damage_id=f'{rec["name"]}_point_{i:02d}',gt_x1=b[0],gt_y1=b[1],gt_x2=b[2],gt_y2=b[3])
   for m in methods:
    v=cache[m][0][i-1];row[f'{m}_rank']=v[0] if v else None;row[f'{m}_score']=(v[1]['score'] if m=='S' else v[1][m]) if v else None
   changes.append(row)
  for m in methods:render(rec,cache[m][1],boxes,OUT/'06_top_candidates'/f'{rec["name"]}_{m}.png',f'{rec["name"]} centroid ranking: {m}; red top10, green GT')
  # coordinate reference results, reusing no score feature change.
  for pos in ('centroid','box_center','peak'):
   for k in (*KS,'all'):
    h=audit.matching(c,boxes,pos,None if k=='all' else k);coordinate.append(dict(image=rec['name'],lighting=light,representative_position=pos,k=k,detected=len(h),point_defects=len(boxes)))
  print(rec['name'],len(c),len(boxes),flush=True)
 # summaries by lighting, split and grid-valid subset
 for subset,rows in [('all',metrics),('evaluation',[x for x in metrics if x['split']=='evaluation']),('grid_valid',[x for x in metrics if next(g['grid_valid_image'] for g in gridrows if g['image']==x['image'])])]:
  for light in ('off','on'):
   for m in sorted({x['feature'] for x in rows if x['lighting']==light}):
    for k in (*KS,'all'):
     z=[x for x in rows if x['lighting']==light and x['feature']==m and x['k']==k];
     if z:summary.append(dict(subset=subset,lighting=light,feature=m,k=k,detected=sum(x['detected'] for x in z),point_defects=sum(x['point_defects'] for x in z),images=len({x['image'] for x in z})))
 write(OUT/'candidate_consistency.csv',consistency);write(OUT/'corrected_ranking_metrics.csv',metrics);write(OUT/'corrected_recall_at_k.csv',summary);write(OUT/'defect_rank_changes.csv',changes);write(OUT/'lighting_summary.csv',[x for x in summary if x['subset']=='all']);write(OUT/'grid_valid_comparison.csv',[x for x in summary if x['subset']=='grid_valid']);write(OUT/'coordinate_comparison.csv',coordinate);write(OUT/'04_grid_valid'/'grid_definition.csv',gridrows)
 bad=[]
 for r in changes:
  a,b=r.get('S_rank'),r.get('D_local_r64_rank')
  if a and b and abs(a-b)>=50:bad.append(r)
 for r in bad[:12]:
  rec=next(x for x in fa.load_records(DATA) if x['name']==r['image']);b=(r['gt_x1'],r['gt_y1'],r['gt_x2'],r['gt_y2']);Image.fromarray(rec['rgb'][max(0,int(b[1])-45):min(rec['height'],int(b[3])+45),max(0,int(b[0])-45):min(rec['width'],int(b[2])+45)]).resize((300,300),Image.Resampling.NEAREST).save(OUT/'08_failure_cases'/f'{r["damage_id"]}.png')
 def val(light,m,k,subset='all'):
  z=[x for x in summary if x['subset']==subset and x['lighting']==light and x['feature']==m and x['k']==k];return f"{z[0]['detected']}/{z[0]['point_defects']}" if z else '—'
 lines=['# 重心座標修正後の点状異常ランキング性能の再評価','', '## 1. 評価座標の修正','候補代表座標は連結成分重心に固定した。外接矩形左上は評価に使わない。候補集合は旧CSVと外接矩形左上・面積・最大Sで照合し、全画像で一致した。Method 0は282,263候補、点状異常27件、重心で25/27件を再現した。','', '## 2. 全21画像のRecall@K（TP/GT）','', '| lighting | feature | @10 | @20 | @50 | @100 | @500 | all |','|---|---|---:|---:|---:|---:|---:|---:|']
 for light in ('off','on'):
  for m in ('S','D_local_r64','Z_local_r64'):
   lines.append('| '+light+' | '+m+' | '+' | '.join(val(light,m,k) for k in (*KS,'all'))+' |')
 lines+=['','## 3. D_gridの公平比較','D_gridは前回定義を維持した。全画像FFTの主周期を求め、重心位置の8x8位相binに属する候補Sの中央値との差 `S - median(S_bin)` とした。FFTピークが弱い画像と低解像度image18は欠損のままとし、0点にはしていない。`grid_valid_comparison.csv` はD_grid有効画像だけの同一分母比較である。','', '## 4. 解釈','ライトなしは4画像・9傷のみなので探索的である。ライトありではD_local/Z_local/D_gridの改善はSと同じ候補集合・同じ傷分母でのみ解釈する。未注釈候補は正常と断定していないため、上位誤反応の種類は人手レビューが必要である。','', '## 5. 結論','主評価では、改善が一貫しなければ最も単純なSを維持する。傷ごとの順位は`defect_rank_changes.csv`、各特徴の上位10候補は`06_top_candidates/`に保存した。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
