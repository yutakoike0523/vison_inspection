"""Conservative FFT ROI experiment; reuses fixed detections, never edits inputs."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw
import fft2_cosine_similarity as fft
import point_anomaly_contrast as pc

ROOT = Path(__file__).resolve().parent
SIZES = (128, 256)
SIM_LIMITS = (.70, .80, .90)
PEAK_LIMITS = (2., 3., 5.)
PARAMS = SimpleNamespace(peak_dc_radius=10., peak_percentile=99., peak_local_max_size=7,
                         peak_radius=150, reference_peak_count=20, peak_min_distance=7.)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_inputs(annotation, previous):
    data = json.loads(annotation.read_text(encoding='utf-8-sig'))
    records, skipped = [], []
    for item in data['images']:
        name = Path(item['imagePath']).stem
        if name not in {f'image{i}' for i in range(1, 22)}:
            continue
        candidates = [annotation.parent / item['imagePath'], ROOT/'data20260903/data20260903'/item['imagePath']]
        path = next((p.resolve() for p in candidates if p.is_file()), None)
        if path is None:
            skipped.append({'image': name, 'reason': 'Missing image'}); continue
        rgb = pc.read_image(path)
        if rgb.shape[:2] != (item['imageHeight'], item['imageWidth']):
            skipped.append({'image': name, 'reason': 'Image/JSON dimension mismatch'}); continue
        boxes = []
        for shape in item['shapes']:
            if shape['label'] != pc.LABEL:
                continue
            if shape['shape_type'] != 'rectangle':
                raise ValueError('Unsupported annotation shape')
            points = np.asarray(shape['points'], float)
            box = [float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]
            assert 0 <= box[0] < box[2] <= rgb.shape[1] and 0 <= box[1] < box[3] <= rgb.shape[0]
            boxes.append(box)
        records.append({'name':name,'path':path,'boxes':boxes,'shape':rgb.shape[:2]})
    records.sort(key=lambda r:(r['name'] not in pc.PILOT,int(r['name'][5:])))
    ds = defaultdict(list)
    for row in read_csv(previous/'detections.csv'):
        if row['family'] != 'background_diff':
            continue
        d = {k:int(row[k]) for k in ['candidate','x1','y1','x2','y2','width','height','area']}
        d.update({k:float(row[k]) for k in ['cx','cy','score','aspect']})
        ds[row['image']].append(d)
    old_manifest = json.loads((previous/'01_original/source_manifest.json').read_text(encoding='utf-8'))
    hashes = {r['image']:r['sha256'] for r in old_manifest}
    for rec in records:
        if hashes.get(rec['name']) != pc.digest(rec['path']):
            raise ValueError(f"Source changed since detection experiment: {rec['name']}")
    cfg = json.loads((previous/'run_config.json').read_text(encoding='utf-8'))
    if cfg['annotation_sha256'] != pc.digest(annotation):
        raise ValueError('Annotation changed since detection experiment')
    return records, ds, skipped


def windows(length, size):
    # Preserve uncovered strips as review rather than extrapolating a classification.
    return range(0, length-size+1, size//2)


def peak_stencil(shape, fx, fy):
    h,w = shape
    ex,ey = w//2+fx*w, h//2+fy*h
    # Six-bin neighborhood must fit and the three-bin signal disk must avoid DC.
    if np.hypot(fx*w,fy*h) <= 4 or ex-6 < 0 or ey-6 < 0 or ex+6 >= w or ey+6 >= h:
        return None
    x0,x1 = int(np.floor(ex-6)), int(np.ceil(ex+6))+1
    y0,y1 = int(np.floor(ey-6)), int(np.ceil(ey+6))+1
    yy,xx = np.mgrid[y0:y1,x0:x1]
    dist = np.hypot(xx-ex, yy-ey)
    dc = np.hypot(xx-w//2,yy-h//2) > 1
    signal = (dist <= 3) & dc
    ring = (dist > 3) & (dist <= 6) & dc
    if signal.sum() < 5 or ring.sum() < 20:
        return None
    return (slice(y0,y1),slice(x0,x1),signal,ring)


def prominence(raw, stencil):
    if stencil is None:
        return None
    sy,sx,signal,ring = stencil
    patch = raw[sy,sx]
    numerator = float(patch[signal].max())
    denominator = float(np.median(patch[ring]))
    # An ill-conditioned ratio is unknown, not evidence that the grid is absent.
    if denominator <= max(float(raw.max())*1e-12, 1e-12):
        return None
    return numerator/denominator


def stable_references(raw):
    refs = fft.reference_peaks(raw, PARAMS)
    pairs = defaultdict(list)
    for ref in refs:
        ratio = prominence(raw,peak_stencil(raw.shape,ref['fx'],ref['fy']))
        pairs[ref['pair_id']].append(dict(ref,global_prominence=ratio))
    eligible = [pair for pair in pairs.values() if len(pair)==2 and
                all(p['global_prominence'] is not None and p['global_prominence']>=3 for p in pair)]
    return [r for pair in eligible for r in pair]


def two_directions(refs):
    unique = {r['pair_id']: np.array([r['fx'],r['fy']]) for r in refs}
    vectors = list(unique.values())
    return any(abs(a[0]*b[1]-a[1]*b[0])/(np.linalg.norm(a)*np.linalg.norm(b)) >= .2
               for i,a in enumerate(vectors) for b in vectors[i+1:])


def features(rec, out):
    name = rec['name']
    gray = fft.read_gray(rec['path'])
    raw, amp = fft.fft_amplitude(gray)
    mask = fft.comparison_mask(129,5)
    global_vector = fft.vector(fft.common_spectrum(amp,129,.125),mask,'log')
    references = stable_references(raw)
    statuses, rows = {}, []
    for n in SIZES:
        pairs=defaultdict(list)
        for ref in references:
            pairs[ref['pair_id']].append((ref,peak_stencil((n,n),ref['fx'],ref['fy'])))
        pairs=[pair for pair in pairs.values() if len(pair)==2 and all(s is not None for _,s in pair)]
        used=[ref for pair in pairs for ref,_ in pair]
        stencils=[s for pair in pairs for _,s in pair]
        stable=two_directions(used)
        statuses[str(n)]={'stable':stable,'pairs':len(pairs),'reason':'usable reference' if stable else 'fewer than two resolved nonparallel conjugate pairs'}
        for y in windows(gray.shape[0],n):
            for x in windows(gray.shape[1],n):
                local_raw, local_amp = fft.fft_amplitude(gray[y:y+n,x:x+n])
                sim = fft.cosine(global_vector,fft.vector(fft.common_spectrum(local_amp,129,.125),mask,'log'))
                ratios=[prominence(local_raw,s) for s in stencils] if stable else []
                valid=bool(ratios) and all(v is not None for v in ratios)
                strength=max(ratios) if valid else None
                rows.append({'image':name,'size':n,'x':x,'y':y,'similarity':sim,
                             'peak_strength':strength,'valid_peak':valid})
        print(f"  {name} {n}px: reference pairs={len(pairs)}, stable={stable}",flush=True)
    pc.write_json(out/'references'/f'{name}.json',{'references':references,'status':statuses})
    pc.csv_out(out/'features'/f'{name}.csv',rows)
    return rows,statuses


def votes(shape, rows, background_flags):
    coverage=np.zeros((shape[0]+1,shape[1]+1),np.int32)
    background=np.zeros_like(coverage)
    for r,bg in zip(rows,background_flags):
        x,y,n=r['x'],r['y'],r['size']
        for yy,xx,s in [(y,x,1),(y+n,x,-1),(y,x+n,-1),(y+n,x+n,1)]:
            coverage[yy,xx]+=s
            background[yy,xx]+=s*int(bg)
    return coverage.cumsum(0).cumsum(1)[:-1,:-1],background.cumsum(0).cumsum(1)[:-1,:-1]


def conservative_mask(shape, rows, similarity_limit, peak_limit):
    excluded=np.ones(shape,bool)
    for size in SIZES:
        part=[r for r in rows if r['size']==size]
        flags=[r['similarity'] is not None and r['similarity']<similarity_limit and r['valid_peak']
               and r['peak_strength']<peak_limit for r in part]
        cover,bg=votes(shape,part,flags)
        excluded &= (cover>0)&(cover==bg)
    return excluded


def legacy_mask(shape, rows, size=256):
    part=[r for r in rows if r['size']==size]
    valid=[r['similarity'] for r in part if r['similarity'] is not None]
    if not valid:
        return np.zeros(shape,bool)
    p20=float(np.percentile(valid,20))
    categories=[2 if r['similarity'] is None else 0 if r['similarity']>=.90 else 1 if r['similarity']<=p20 else 2 for r in part]
    counts=[]
    for category in range(3):
        cover,count=votes(shape,part,[c==category for c in categories])
        counts.append(count)
    # Exactly the old class order: grid > background > review in a voting tie.
    return (np.argmax(np.stack(counts),axis=0)==1)&(cover>0)


def retention(box, excluded):
    x1,y1,x2,y2=box
    left,top=int(np.floor(x1)),int(np.floor(y1))
    right,bottom=int(np.ceil(x2)),int(np.ceil(y2))
    # Fractional rectangle area, not a rounded approximation for tiny image18 GTs.
    wx=np.maximum(0,np.minimum(np.arange(left,right)+1,x2)-np.maximum(np.arange(left,right),x1))
    wy=np.maximum(0,np.minimum(np.arange(top,bottom)+1,y2)-np.maximum(np.arange(top,bottom),y1))
    removed=float(np.sum(excluded[top:bottom,left:right]*wy[:,None]*wx[None,:]))
    center_kept=not bool(excluded[int((y1+y2)/2),int((x1+x2)/2)])
    return center_kept,1-removed/((x2-x1)*(y2-y1))


def filter_detections(ds, excluded):
    return [d for d in ds if not excluded[int(d['cy']),int(d['cx'])]]


def sweep(records, all_features, detections, out):
    result=[]
    for s in SIM_LIMITS:
        for p in PEAK_LIMITS:
            centers=0; areas=[]; fp=0; removed=0; lost=0
            for rec in records:
                if rec['name'] not in pc.PILOT:
                    continue
                ex=conservative_mask(rec['shape'],all_features[rec['name']],s,p)
                original=detections[rec['name']]
                ds=filter_detections(original,ex)
                baseline,_=pc.match(original,rec['boxes'],'center')
                matched,_=pc.match(ds,rec['boxes'],'center')
                lost+=len(set(baseline.values())-set(matched.values()))
                fp+=len(ds)-len(matched);removed+=int(ex.sum())
                for box in rec['boxes']:
                    center,area=retention(box,ex);centers+=int(center);areas.append(area)
            result.append({'similarity_limit':s,'peak_limit':p,'gt':len(areas),'centers_retained':centers,
                'minimum_gt_area_retained':min(areas) if areas else 0.,
                'eligible':len(areas)==13 and centers==13 and min(areas)>=.95,
                'fp':fp,'excluded_pixels':removed,'previous_tp_lost':lost})
    pc.csv_out(out/'pilot_parameter_sweep.csv',result)
    eligible=[r for r in result if r['eligible']]
    # FP ties: fewer previously detected GT lost, then less area removed, then lower thresholds.
    selected=min(eligible,key=lambda r:(r['fp'],r['previous_tp_lost'],r['excluded_pixels'],r['similarity_limit'],r['peak_limit'])) if eligible else None
    pc.write_json(out/'selected_parameters.json',{'selected':selected,'fallback':'no exclusion' if selected is None else None})
    return selected


def color_mask(rgb, excluded):
    result=rgb.copy()
    result[excluded]=(rgb[excluded]*.45+np.array([255,80,0])*.55).astype(np.uint8)
    return result


def evaluate(records, all_features, detections, selected, out):
    image_metrics=[]; gt_rows=[]; det_rows=[]; inspection=[]
    for rec in records:
        name=rec['name'];rgb=pc.read_image(rec['path']);boxes=rec['boxes']
        ex_new=conservative_mask(rec['shape'],all_features[name],selected['similarity_limit'],selected['peak_limit']) if selected else np.zeros(rec['shape'],bool)
        masks={'none':np.zeros(rec['shape'],bool),'legacy':legacy_mask(rec['shape'],all_features[name]),'proposed':ex_new}
        pc.save_image(out/'masks'/f'{name}_legacy128_excluded.png',legacy_mask(rec['shape'],all_features[name],128).astype(np.uint8)*255)
        panels=[];overlays=[]
        old_assignments={c:set(pc.match(detections[name],boxes,c)[0].values()) for c in ['center','iou']}
        for method,ex in masks.items():
            ds=filter_detections(detections[name],ex)
            kept_ids={d['candidate'] for d in ds}
            match_ids={}
            retention_values=[retention(b,ex) for b in boxes]
            for criterion in ['center','iou']:
                assignment,ious=pc.match(ds,boxes,criterion)
                match_ids[criterion]={ds[j]['candidate']:(gt+1,ious[j]) for j,gt in assignment.items()}
                tp,fp,fn=len(assignment),len(ds)-len(assignment),len(boxes)-len(assignment)
                image_metrics.append({'image':name,'split':'pilot' if name in pc.PILOT else 'remaining',
                    'method':method,'criterion':criterion,**pc.summarize(tp,fp,fn,1),
                    'excluded_pixels':int(ex.sum()),'pixels':ex.size,'excluded_fraction':float(ex.mean()),
                    'centers_retained':sum(c for c,a in retention_values),
                    'rectangles_95pct_retained':sum(a>=.95 for c,a in retention_values),
                    'previous_tp_lost':len(old_assignments[criterion]-set(assignment.values()))})
            for gt,(box,(center,area)) in enumerate(zip(boxes,retention_values)):
                gt_rows.append({'image':name,'gt':gt+1,'method':method,'center_retained':center,'area_retained':area})
            for d in detections[name]:
                cm=match_ids['center'].get(d['candidate'],('', ''))
                im=match_ids['iou'].get(d['candidate'],('', ''))
                det_rows.append({'image':name,'method':method,'candidate':d['candidate'],'kept':d['candidate'] in kept_ids,
                                'x1':d['x1'],'y1':d['y1'],'x2':d['x2'],'y2':d['y2'],'score':d['score'],
                                'center_gt':cm[0],'center_iou':cm[1],'iou_gt':im[0],'iou':im[1]})
            pc.save_image(out/'masks'/f'{name}_{method}_excluded.png',ex.astype(np.uint8)*255)
            panel=pc.draw_boxes(color_mask(rgb,ex),boxes)
            overlay=pc.draw_boxes(color_mask(rgb,ex),boxes,ds)
            panels.append(panel);overlays.append(overlay)
            pc.save_image(out/'detections'/f'{name}_{method}.png',overlay)
        titles=['No exclusion','Legacy 256px: .90 / bottom20%','Proposed: similarity + grid peaks']
        pc.montage(panels,titles,width=640).save(out/'comparison'/f'{name}_masks.png')
        pc.montage(overlays,titles,width=640).save(out/'comparison'/f'{name}_detections.png')
        for gt,box in enumerate(boxes):
            cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            side=max(64,int(np.ceil(max(box[2]-box[0],box[3]-box[1])+40)))
            x1,y1=max(0,int(cx-side/2)),max(0,int(cy-side/2))
            x2,y2=min(rgb.shape[1],x1+side),min(rgb.shape[0],y1+side)
            arrays=[rgb[y1:y2,x1:x2]]+[p[y1:y2,x1:x2] for p in overlays]
            pc.montage(arrays,['Original pixels']+titles,width=4*(x2-x1),nearest=True).save(out/'crops'/f'{name}_gt{gt+1:02d}.png')
        # Review-only windows: never used to construct or tune a mask.
        rois={'image8':{'tag_center':(300,350,650,700),'tag_edge':(650,700,850,900),
                        'outside':(1000,1250,1800,1536),'texture':(900,100,1300,500)},
              'image16':{'tag_center':(1870,500,2048,950),'tag_edge':(1650,550,1850,950)},
              'image20':{'tag_center':(1830,600,2048,1000),'tag_edge':(1600,550,1800,950)}}
        for label,coords in rois.get(name,{}).items():
            x1,y1,x2,y2=coords
            pc.montage([rgb[y1:y2,x1:x2]]+[p[y1:y2,x1:x2] for p in overlays],['Original']+titles,width=400).save(out/'review'/f'{name}_{label}.png')
            inspection.append({'image':name,'region':label,'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                               **{m+'_excluded_fraction':float(ex[y1:y2,x1:x2].mean()) for m,ex in masks.items()}})
        print('Evaluated:',name,flush=True)
    metrics=[]
    for method in ['none','legacy','proposed']:
        for split in ['pilot','remaining','all']:
            for criterion in ['center','iou']:
                rows=[r for r in image_metrics if r['method']==method and r['criterion']==criterion and (split=='all' or r['split']==split)]
                sums=[sum(r[k] for r in rows) for k in ['tp','fp','fn','images']]
                metrics.append({'method':method,'split':split,'criterion':criterion,**pc.summarize(*sums),
                    **{k:sum(r[k] for r in rows) for k in ['centers_retained','rectangles_95pct_retained','previous_tp_lost']},
                    'excluded_fraction':sum(r['excluded_pixels'] for r in rows)/sum(r['pixels'] for r in rows)})
    pc.csv_out(out/'metrics.csv',metrics);pc.csv_out(out/'metrics_per_image.csv',image_metrics)
    pc.csv_out(out/'gt_retention.csv',gt_rows);pc.csv_out(out/'detections.csv',det_rows)
    pc.csv_out(out/'review/regions.csv',inspection)
    return metrics


def report(out,metrics,selected,statuses):
    rows=['# FFT類似度と格子ピークによる背景除外','', '## 目的','',
        '傷を残すことを優先し、札・対象外領域を検出対象から除外できるかを検証する。FFTは領域選別だけに使用する。','',
        '## 手法','',
        '前回のbackground_diff（sigma=16、暗点しきい値40、moderateサイズ制約）の候補をCSVから固定して読み込む。候補矩形中心の画素が除外マスク内なら候補を除く。画像をマスクして再検出することはせず、人工的な境界や候補分裂を生じさせない。',
        '元画像・注釈のハッシュが前回と一致しなければ停止する。全21画像、点状異常27件を対象とする。image18だけ428×321 px、他は2048×1536 px。元解像度を維持する。',
        '既存FFT関数による平均除去＋Hann窓、振幅スペクトルのlog1pコサイン類似度。共通グリッド129×129、±0.125 cycles/pixel、DC中心±5点を除外。128/256 px、50%オーバーラップ、画像端の未被覆部分は保留。',
        '全体FFTから既存既定値で最大20個の共役ピークを抽出。基準の強さは3倍以上を両側で要求。局所FFTで半径3 binの信号円がDCから離れ、外側半径6 binの評価範囲が収まる共役対を使用。方向の正弦が0.2以上となる2対以上がなければ、その尺度は保留。これは参照安定性の保守的な代理条件であり、格子を保証するものではない。',
        'ピークの強さ＝基準周波数の半径3 bin内最大振幅／外側3–6 bin環状域の振幅中央値（DC半径1 binを除外）。分母が数値的にゼロなら保留。他のピークが環状域に入る影響はあり得る。全参照ピークの最大値を採用し、片側1点でも強ければ背景除外を控える。',
        '新方式は類似度<Sかつピーク強さ<Pのブロックを背景候補とする。128・256の両尺度それぞれで、その画素を覆う全ブロックが背景候補の場合だけ除外。無効ブロックは背景への投票をせず除外を防ぐ。',
        '現行方式は256 pxで、類似度≥0.90を格子、未満かつ画像内p20以下を背景、それ以外を保留。重複ブロックの多数決。同数の場合は既存処理通り格子→背景→保留の順。128 px現行マスクは参考保存するが主比較には使わない。',
        '予備image16/18/20でS=0.70/0.80/0.90、P=2/3/5の9条件を比較。13件全ての中心と各矩形の95%以上が残る条件だけを候補とし、未対応候補数を最小化。同点なら既検出正解の喪失数、除外面積、S、Pが小さい順。条件なしなら新方式は除外なしに戻す。残り18画像では再調整しない。',
        '正解矩形の保持面積は小数座標と画素セルとの交差面積で算出。検出対応は前回同様、中心包含とIoU≥0.10を別々に最大対応数優先の1対1割当で集計。previous_tp_lostは除外前に対応した正解IDのうち対応が失われた件数。',
        '図は橙=除外領域、緑=正解矩形、マゼンタ=採用候補。cropsは元画素を含む同座標4倍最近傍比較。札中央・札縁等のreview切り出しは目視確認用の矩形であり、正解セグメンテーションではなく、条件選択にも使わない。','',
        '## 結果','',f'選択条件: `{json.dumps(selected,ensure_ascii=False)}`','',
        '|方式|区分|基準|TP/GT|Precision %|Recall %|F1|FP/枚|中心保持|面積95%保持|既検出喪失|除外面積 %|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in metrics:
        rows.append(f"|{r['method']}|{r['split']}|{r['criterion']}|{r['tp']}/{r['gt']}|{100*r['precision']:.4f}|{100*r['recall']:.2f}|{r['f1']:.6f}|{r['fp_per_image']:.2f}|{r['centers_retained']}|{r['rectangles_95pct_retained']}|{r['previous_tp_lost']}|{100*r['excluded_fraction']:.2f}|")
    rows+=['','参照判定（尺度ごとの有効な共役対数と保留理由）は `references/*.json`。', '',
           '|画像|128px安定|256px安定|','|---|---|---|']
    for name,s in statuses.items():
        rows.append(f"|{name}|{s['128']['stable']} ({s['128']['pairs']} pairs)|{s['256']['stable']} ({s['256']['pairs']} pairs)|")
    lookup={(r['method'],r['split'],r['criterion']):r for r in metrics}
    original=lookup['none','all','center'];old=lookup['legacy','all','center'];new=lookup['proposed','all','center']
    reduction=1-new['fp']/original['fp'] if original['fp'] else 0.
    rows+=['','## 考察','',
        f"新方式は未対応候補を{reduction:.2%}削減し、中心保持は{new['centers_retained']}/{new['gt']}件、各矩形95%以上の保持は{new['rectangles_95pct_retained']}/{new['gt']}件。現行方式の中心保持は{old['centers_retained']}/{old['gt']}件、95%保持は{old['rectangles_95pct_retained']}/{old['gt']}件。",
        f"現行方式の未対応候補削減率は{1-old['fp']/original['fp']:.2%}。既検出の正解対応を{old['previous_tp_lost']}件失った。今回除外された正解領域はimage16 GT2、image18 GT3/GT4、image20 GT2（IDは画像内の点状異常順）。`crops/`で除外前後を確認できる。",
        '今回の予備9条件は全て除外0画素・未対応候補30件で同点だった。選択されたS=0.70/P=2は同点規則による形式的な選択であり、この値が背景分離に優れるという証拠ではない。image16は128pxで有効な参照対が0、image20は128pxで1対かつ256pxでも非平行な2方向の条件を満たさず、保留になった。',
        '予備の保持条件は残り画像での保証ではない。全画像で保持条件を割る場合は採用を推奨せず、そのまま失敗例として報告する。多数の候補が残る状況では偶然正解内に中心が入る可能性もあり、Recallだけでは性能を示せない。',
        '新方式は両尺度・全被覆ブロックで一致を要求するため、傷を残しやすい反面、格子と札が同居するブロックでは札が残る。現行方式との違いには特徴量だけでなく空間的な合意条件も含まれ、削減率の差をピーク特徴単体の効果とは解釈しない。',
        '格子参照の不安定な画像を保留すると除外量は減る。FFTピークには格子以外の繰り返しやエッジも混ざり得る。画像18の解像度差や画像内の周期変化に対して物理スケールを正規化していない。',
        '特に128pxでは低い格子周波数がDC近傍に寄り、既存の±3 bin許容範囲とDC回避条件を同時に満たしにくい。ピーク周辺最大値を複数参照の最大値で集約する条件も保守的で、格子以外の強い成分でも除外を止める。これらは今回の固定設定の制約であり、FFTによる領域選別一般が不可能という結果ではない。`feature_summary.csv`に各尺度の有効ブロック数、強さの分布、背景条件成立数を記録する。',
        '背景・札の正解マスクがないため背景除外の正解率は算出できない。未対応候補には未注釈の傷や他ラベルがあり得るためPrecision・FPは暫定値。検査対象内部の粒状模様による誤検出は領域除外だけでは解決しない。','',
        '## 結論','']
    if new['rectangles_95pct_retained']<new['gt'] or new['centers_retained']<new['gt']:
        rows+=['新方式は全画像で傷保持の目標を満たさなかった。予備画像の選択条件をそのまま運用へ採用することは推奨しない。']
    elif reduction<=0:
        rows+=['傷は保持したが、今回の保守的な条件では未対応候補の削減効果を確認できなかった。現行方式より有効とする根拠は得られていない。']
    else:
        rows+=[f'新方式は今回の27件について保持条件を満たし、未対応候補を{reduction:.2%}削減した。ただし背景の正解マスクと独立画像での検証は未実施であり、背景除外が完成したとはいえない。']
    rows+=['','再実行: `python point_anomaly_fft_mask.py`。元画像・前回成果物は読み取り専用で参照する。`verification.json`にハッシュと集計検証を記録。']
    # Keep prose paragraphs separate while preserving tables.
    formatted=[]
    for row in rows:
        if formatted and row and formatted[-1] and not row.startswith(('|','#')) and not formatted[-1].startswith(('|','#')):
            formatted.append('')
        formatted.append(row)
    (out/'report.md').write_text('\n'.join(formatted)+'\n',encoding='utf-8')


def feature_summary(all_features, selected, out):
    rows=[]
    for name,features_for_image in all_features.items():
        for size in SIZES:
            part=[r for r in features_for_image if r['size']==size]
            valid=[r for r in part if r['valid_peak']]
            strength=[r['peak_strength'] for r in valid]
            sim=selected['similarity_limit'] if selected else .7
            peak=selected['peak_limit'] if selected else 2.
            rows.append({'image':name,'size':size,'blocks':len(part),'valid_peak_blocks':len(valid),
                'peak_strength_min':min(strength) if strength else '',
                'peak_strength_median':float(np.median(strength)) if strength else '',
                'low_similarity_blocks':sum(r['similarity'] is not None and r['similarity']<sim for r in part),
                'weak_peak_blocks':sum(r['peak_strength']<peak for r in valid),
                'background_blocks':sum(r['similarity'] is not None and r['similarity']<sim and r['peak_strength']<peak for r in valid)})
    pc.csv_out(out/'feature_summary.csv',rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--annotation',type=Path,default=ROOT/'data20260903/data20260903/全傷.json')
    p.add_argument('--previous',type=Path,default=ROOT/'outputs/point_anomaly_contrast')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/point_anomaly_fft_mask')
    args=p.parse_args();out=args.output
    if out.resolve()==args.previous.resolve() or args.previous.resolve() in out.resolve().parents:
        raise ValueError('Output must be separate from previous artifacts')
    cv2.setNumThreads(4)
    records,detections,skipped=load_inputs(args.annotation,args.previous)
    tracked=[r['path'] for r in records]+[args.annotation]+[f for f in args.previous.rglob('*') if f.is_file()]
    hashes={str(f.resolve()):pc.digest(f) for f in tracked}
    for folder in ['references','features','masks','comparison','detections','crops','review']:
        (out/folder).mkdir(parents=True,exist_ok=True)
    pc.write_json(out/'skipped.json',skipped)
    all_features={};statuses={}
    # Pilot is processed first; selection must precede remaining-image evaluation.
    for rec in records:
        if rec['name'] in pc.PILOT:
            all_features[rec['name']],statuses[rec['name']]=features(rec,out)
    selected=sweep([r for r in records if r['name'] in pc.PILOT],all_features,detections,out)
    print('Locked pilot selection:',selected,flush=True)
    for rec in records:
        if rec['name'] not in pc.PILOT:
            all_features[rec['name']],statuses[rec['name']]=features(rec,out)
    metrics=evaluate(records,all_features,detections,selected,out)
    feature_summary(all_features,selected,out)
    report(out,metrics,selected,statuses)
    previous_metrics=read_csv(args.previous/'metrics.csv')
    for row in metrics:
        if row['method']!='none':continue
        prior=next(r for r in previous_metrics if r['family']=='background_diff' and r['split']==row['split'] and r['criterion']==row['criterion'])
        assert all(row[k]==int(prior[k]) for k in ['tp','fp','fn','gt','images'])
    assert all(pc.digest(Path(f))==h for f,h in hashes.items()), 'Input artifact was modified'
    pc.write_json(out/'run_config.json',{'sizes':SIZES,'similarity_limits':SIM_LIMITS,'peak_limits':PEAK_LIMITS,
        'reference_parameters':vars(PARAMS),'pilot':sorted(pc.PILOT),'detector':json.loads((args.previous/'selected_parameters.json').read_text())['background_diff'],
        'selected':selected,'source_hashes':hashes,'implementation_sha256':pc.digest(Path(__file__))})
    pc.write_json(out/'verification.json',{'images':len(records),'gt':sum(len(r['boxes']) for r in records),
        'all_original_and_previous_artifact_hashes_unchanged':True,'unchanged_files':len(hashes),'no_mask_metrics_equal_previous':True})
    print('Complete:',out,flush=True)


if __name__=='__main__':main()
