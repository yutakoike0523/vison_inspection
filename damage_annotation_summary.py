"""Summarize annotated point, line, area, and local-deformation anomalies."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

import fft_mask_ground_truth_evaluation as gt
from background_surface_hist_experiment import write_csv

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data20260903"/"data20260903"
OUT=ROOT/"outputs"/"damage_annotation_summary"
# Values follow the existing DAMAGE_MAP insertion order; dent normalizes to local deformation.
CODES={list(dict.fromkeys(gt.DAMAGE_MAP.values()))[0]:"point",list(dict.fromkeys(gt.DAMAGE_MAP.values()))[1]:"line",list(dict.fromkeys(gt.DAMAGE_MAP.values()))[2]:"area",list(dict.fromkeys(gt.DAMAGE_MAP.values()))[3]:"local_deformation"}

def percentile(a,q): return float(np.percentile(a,q)) if len(a) else None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,default=DATA); ap.add_argument("--output",type=Path,default=OUT); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for jp in sorted(args.data.glob("image*.json"),key=lambda p:int(p.stem[5:])):
        rec=gt.load_record(jp)
        for shape in rec["shapes"]:
            if shape["normalized_label"] not in CODES or shape["box"] is None: continue
            x1,y1,x2,y2=shape["box"]; w=x2-x1; h=y2-y1
            rows.append({"image":rec["name"],"category":CODES[shape["normalized_label"]],"source_label":shape["label"],"x1":x1,"y1":y1,"x2":x2,"y2":y2,"width_px":w,"height_px":h,"area_px2":w*h,"aspect_ratio_long_short":max(w,h)/max(1e-9,min(w,h)),"center_x":(x1+x2)/2,"center_y":(y1+y2)/2,"image_width":rec["width"],"image_height":rec["height"],"area_percent_of_image":100*w*h/(rec["width"]*rec["height"])})
    write_csv(args.output/"damage_instances.csv",rows)
    summary=[]
    for category in ("point","line","area","local_deformation"):
        data=[r for r in rows if r["category"]==category]
        summary.append({"category":category,"count":len(data),"image_count":len({r["image"] for r in data}),"width_median_px":percentile([r["width_px"] for r in data],50),"height_median_px":percentile([r["height_px"] for r in data],50),"area_median_px2":percentile([r["area_px2"] for r in data],50),"area_mean_px2":float(np.mean([r["area_px2"] for r in data])) if data else None,"area_p90_px2":percentile([r["area_px2"] for r in data],90),"aspect_median":percentile([r["aspect_ratio_long_short"] for r in data],50),"largest_instance":max(data,key=lambda r:r["area_px2"])["image"]+" "+max(data,key=lambda r:r["area_px2"])["source_label"] if data else ""})
    write_csv(args.output/"damage_category_summary.csv",summary)
    by_image=[]
    for image in sorted({r["image"] for r in rows},key=lambda x:int(x[5:])):
        subset=[r for r in rows if r["image"]==image]
        by_image.append({"image":image,**{c:sum(r["category"]==c for r in subset) for c in ("point","line","area","local_deformation")},"total":len(subset),"total_area_px2":sum(r["area_px2"] for r in subset)})
    write_csv(args.output/"damage_by_image.csv",by_image)
    lines=["# Damage annotation summary","","The categories are the existing Labelme rectangle annotations. Counts describe annotations, not automatically detected defects.","","| category | count | images | median W x H px | median area px2 | median aspect |","|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['category']} | {r['count']} | {r['image_count']} | {r['width_median_px']:.1f} x {r['height_median_px']:.1f} | {r['area_median_px2']:.1f} | {r['aspect_median']:.2f} |" for r in summary]
    lines += ["","## Notes","- point: compact local brightness anomaly; line: elongated anomaly; area: broad irregular region; local_deformation: dent/deformation annotation.","- Rectangle extent is the annotation bounding box, so it may include unaffected pixels around the defect.","- See damage_instances.csv for image, position, dimensions, area, and aspect ratio of every annotation."]
    (args.output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"instances={len(rows)} output={args.output}")
if __name__=="__main__":main()
