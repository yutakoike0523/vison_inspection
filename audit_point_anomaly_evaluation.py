"""Audit the Method-0 25/27 vs 20/27 discrepancy without changing detection."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import point_anomaly_feature_analysis as fa

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data20260903'/'data20260903'
OUT=ROOT/'outputs'/'point_anomaly_evaluation_audit'
PRIOR=ROOT/'outputs'/'point_anomaly_peak_analysis'/'09_gaussian_s8_top10'/'all_candidate_scores.csv'
LIGHT_OFF={'image1','image19','image20','image21'}; FOCUS={'image3','image5','image8','image10','image17'}
THRESH=20.294957160949707; KS=(10,20,50,100,500)

def write_csv(p,rows):
    rows=list(rows);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
def pts(rec):return [s for s in rec['shapes'] if s['normalized_label']=='点状異常']
def cand(rec):
    gray=rec['gray'].astype(np.float32);r=cv2.GaussianBlur(gray,(0,0),8,borderType=cv2.BORDER_REFLECT_101)-gray
    # Explicitly request the same 8-connectivity, using the fast Grana CCL
    # backend.  It changes no foreground pixel or component definition.
    n,lab,stats,cent=cv2.connectedComponentsWithStatsWithAlgorithm((r>=THRESH).astype(np.uint8),8,cv2.CV_32S,cv2.CCL_GRANA);out=[]
    for k in range(1,n):
        x,y,w,h,a=map(int,stats[k]);
        if not(2<=a<=2000 and w<=100 and h<=100):continue
        # Restrict each maximum search to its stats bounding box.  Scanning the
        # full image for every label is equivalent but needlessly O(labels*pixels).
        roi_r=r[y:y+h,x:x+w]; roi_lab=lab[y:y+h,x:x+w];masked=np.where(roi_lab==k,roi_r,-np.inf);iy,ix=np.unravel_index(np.argmax(masked),masked.shape);px,py=x+int(ix),y+int(iy)
        out.append(dict(component_id=len(out)+1,x=x,y=y,w=w,h=h,area=a,cx=float(cent[k,0]),cy=float(cent[k,1]),peak_x=px,peak_y=py,score=float(masked[iy,ix])))
    return sorted(out,key=lambda q:(-q['score'],q['y'],q['x']))
def inside(x,y,b):return b[0]<=x<=b[2] and b[1]<=y<=b[3]
def overlap(c,b):
    x1=max(c['x'],b[0]);y1=max(c['y'],b[1]);x2=min(c['x']+c['w'],b[2]);y2=min(c['y']+c['h'],b[3]);return max(0,x2-x1)*max(0,y2-y1)
def distance(c,b):
    x,y=c['cx'],c['cy'];dx=max(b[0]-x,0,x-b[2]);dy=max(b[1]-y,0,y-b[3]);return math.hypot(dx,dy)
def matching(cands,boxes,position,k=None):
    chosen=cands[:k] if k else cands;hit=set()
    for c in chosen:
        x,y=(c['cx'],c['cy']) if position=='centroid' else ((c['x']+c['w']/2,c['y']+c['h']/2) if position=='box_center' else (c['peak_x'],c['peak_y']) if position=='peak' else (c['x'],c['y']))
        for i,b in enumerate(boxes):
            if i not in hit and inside(x,y,b):hit.add(i);break
    return hit
def prior_rows():
    d={}
    with PRIOR.open(encoding='utf-8-sig') as f:
        for z in csv.DictReader(f):d.setdefault(z['image'],[]).append(z)
    return d
def render(rec,c,b,prior_tp,current_tp,path):
    im=Image.fromarray(rec['rgb'].copy());d=ImageDraw.Draw(im);line=max(2,rec['width']//700)
    d.rectangle(b,outline='lime',width=line)
    nearest=sorted(c,key=lambda q:distance(q,b))[:5]
    for j,q in enumerate(nearest,1):
        d.rectangle((q['x'],q['y'],q['x']+q['w'],q['y']+q['h']),outline='blue',width=line)
        d.ellipse((q['cx']-3,q['cy']-3,q['cx']+3,q['cy']+3),outline='orange',width=line)
        d.ellipse((q['peak_x']-3,q['peak_y']-3,q['peak_x']+3,q['peak_y']+3),outline='purple',width=line)
        d.text((q['x'],max(0,q['y']-14)),str(j),fill='white')
    note=f'prior centroid TP={prior_tp}; current bbox-left TP={current_tp}; blue bbox, orange centroid, purple peak'
    d.rectangle((3,3,min(rec['width']-3,1050),35),fill='black');d.text((8,10),note,fill='white');im.save(path)
    # crop around GT and closest candidates, without changing source pixels.
    xs=[b[0],b[2]]+[q['x'] for q in nearest]+[q['x']+q['w'] for q in nearest];ys=[b[1],b[3]]+[q['y'] for q in nearest]+[q['y']+q['h'] for q in nearest];pad=50
    x1=max(0,int(min(xs)-pad));x2=min(rec['width'],int(max(xs)+pad));y1=max(0,int(min(ys)-pad));y2=min(rec['height'],int(max(ys)+pad));im.crop((x1,y1,x2,y2)).resize(((x2-x1)*3,(y2-y1)*3),Image.Resampling.NEAREST).save(path.with_name(path.stem+'_crop.png'))
def main():
    for p in (OUT,OUT/'failure_cases'):p.mkdir(parents=True,exist_ok=True)
    old=prior_rows();comparison=[];coords=[];recheck=[];consistency=[]
    for rec in fa.load_records(DATA):
        c=cand(rec);boxes=[s['box'] for s in pts(rec)]; prior=old.get(rec['name'],[])
        # Old all-candidate CSV intentionally stores bbox left/top, not centroid.
        key_new={(q['x'],q['y'],round(q['score'],6),q['area']) for q in c};key_old={(int(float(q['component_x'])),int(float(q['component_y'])),round(float(q['score']),6),int(float(q['area']))) for q in prior}
        consistency.append(dict(image=rec['name'],prior_candidates=len(prior),recomputed_candidates=len(c),shared=len(key_old&key_new),prior_only=len(key_old-key_new),recomputed_only=len(key_new-key_old)))
        hit_cent=matching(c,boxes,'centroid');hit_left=matching(c,boxes,'left');hit_box=matching(c,boxes,'box_center');hit_peak=matching(c,boxes,'peak')
        for i,b in enumerate(boxes,1):
            near=sorted(c,key=lambda q:distance(q,b))[:5]; best=near[0]
            prior_tp=i-1 in hit_cent; current_tp=i-1 in hit_left
            reason='same result'
            if prior_tp and not current_tp:
                reason='evaluation-position bug: prior CSV bbox left/top was used as candidate coordinate instead of connected-component centroid'
            elif not prior_tp: reason='not detected by centroid under common Method-0 candidate set'
            comparison.append(dict(image=rec['name'],lighting='off' if rec['name'] in LIGHT_OFF else 'on',damage_id=f'{rec["name"]}_point_{i:02d}',prior_TP=int(prior_tp),current_TP_bbox_left_top=int(current_tp),centroid_TP=int(prior_tp),box_center_TP=int(i-1 in hit_box),peak_TP=int(i-1 in hit_peak),reason=reason))
            for rank,q in enumerate(near,1):coords.append(dict(image=rec['name'],damage_id=f'{rec["name"]}_point_{i:02d}',gt_x1=b[0],gt_y1=b[1],gt_x2=b[2],gt_y2=b[3],nearest_rank=rank,component_id=q['component_id'],component_x=q['x'],component_y=q['y'],component_w=q['w'],component_h=q['h'],bbox_center_x=q['x']+q['w']/2,bbox_center_y=q['y']+q['h']/2,centroid_x=q['cx'],centroid_y=q['cy'],peak_x=q['peak_x'],peak_y=q['peak_y'],score=q['score'],area=q['area'],bbox_left_top_in_gt=int(inside(q['x'],q['y'],b)),bbox_center_in_gt=int(inside(q['x']+q['w']/2,q['y']+q['h']/2,b)),centroid_in_gt=int(inside(q['cx'],q['cy'],b)),peak_in_gt=int(inside(q['peak_x'],q['peak_y'],b)),bbox_gt_overlap_px=overlap(q,b),centroid_to_gt_distance_px=distance(q,b)))
            if rec['name'] in FOCUS and prior_tp and not current_tp:render(rec,c,b,prior_tp,current_tp,OUT/'failure_cases'/f'{rec["name"]}_point_{i:02d}.png')
        for pos in ('centroid','box_center','peak','left'):
            for k in (*KS,'all'):
                h=matching(c,boxes,pos,None if k=='all' else k);recheck.append(dict(image=rec['name'],lighting='off' if rec['name'] in LIGHT_OFF else 'on',resolution_group='low' if rec['name']=='image18' else 'standard',representative_position=pos,k=k,point_defects=len(boxes),detected=len(h),candidate_count=len(c)))
        print(rec['name'],len(c),len(hit_cent),len(hit_left),flush=True)
    write_csv(OUT/'comparison.csv',comparison);write_csv(OUT/'coordinate_comparison.csv',coords);write_csv(OUT/'evaluation_recheck.csv',recheck);write_csv(OUT/'candidate_set_consistency.csv',consistency)
    diff=[r for r in comparison if r['prior_TP'] and not r['current_TP_bbox_left_top']]
    lines=['# Gaussian背景差分 Method 0 検出数不一致の監査','',f'## 結論','同じ282,263候補で **25/27から20/27へ減った原因は検出器ではなく、評価に使った候補代表位置の取り違え** だった。前回CSVの `component_x/component_y` は外接矩形の左上座標である。今回の集計がこれを連結成分重心として扱ったため、重心は正解矩形内でも左上は外にある5件を偽陰性にした。','',f'- 前回（再計算した重心規則）: {sum(r["prior_TP"] for r in comparison)}/{len(comparison)}',f'- 今回の誤った左上規則: {sum(r["current_TP_bbox_left_top"] for r in comparison)}/{len(comparison)}','- 差分: '+', '.join(r['damage_id'] for r in diff),'','## 候補集合の照合','`candidate_set_consistency.csv` で、外接矩形左上・面積・最大scoreの組を突き合わせた。全画像で差分が0なら、候補集合・Gaussian・しきい値・8近傍連結成分・形状フィルタは一致している。','', '## 評価規則','従来の共通規則は **各連結成分の重心が、大小を正規化したLabelme矩形に入るか** とする。順位評価はscore降順、同点はy,x順、同じ傷には最初の1候補だけを対応させる。tag/backgroundは主評価でマスクしない。','', '## 可視化','重点5画像で、差分傷があるものを `failure_cases/` に保存した。緑=正解矩形、青=候補外接矩形、橙=重心、紫=最大応答位置。最も近い5候補を表示し、拡大画像も同名`_crop.png`で保存した。','', '## 代表位置比較','`evaluation_recheck.csv` に重心、外接矩形中心、最大応答位置、左上のRecall@10/20/50/100/500/allを、ライト別・画像別で保存した。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
