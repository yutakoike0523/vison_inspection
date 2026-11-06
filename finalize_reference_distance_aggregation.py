#!/usr/bin/env python3
"""Create plots, settings, and Japanese report from completed experiment CSV files."""
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import reference_distance_aggregation_experiment as exp


def value(s):
    if s == "": return None
    if s in ("True", "False"): return s == "True"
    try: return float(s)
    except ValueError: return s


def read(path): return [{k:value(v) for k,v in r.items()} for r in csv.DictReader(path.open(encoding="utf-8-sig"))]


def fmt(x, spec=".4f"): return format(x, spec) if x is not None and isinstance(x,(int,float)) and np.isfinite(x) else "NA"


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=exp.OUT);args=p.parse_args();o=args.output
    overall=read(o/"overall_summary.csv");types=read(o/"per_type_metrics.csv");cont=read(o/"reference_contamination.csv");timing=read(o/"processing_time.csv")
    A=[r for r in overall if r.get("experiment")=="A" and r.get("scope")=="common_region"]
    B=[r for r in overall if r.get("experiment")=="B" and r.get("scope")=="common_region"]
    Cf=[r for r in overall if r.get("experiment")=="C_fixed" and r.get("scope")=="common_region"]
    Cs=[r for r in overall if r.get("experiment")=="C_selected" and r.get("scope")=="common_region"]
    cor=[r for r in overall if r.get("aggregation_level")=="contamination_correlation"]
    bestA=max(A,key=lambda r:r["pr_auc"]);bestB=max(B,key=lambda r:r["pr_auc"]);bestCs=max(Cs,key=lambda r:r["pr_auc"])
    selected_mode,selected_distance=Cs[0]["window_mode"],Cs[0]["distance_name"]
    # Metric figures.
    orderA=[f"d{d}" for d in exp.A_DISTANCES];A.sort(key=lambda r:orderA.index(r["distance_name"]));x=np.array(exp.A_DISTANCES)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,key,title in zip(axes,("roc_auc","pr_auc","defect_detection_rate_fpr1"),("ROC-AUC","PR-AUC","Defect detection @ FPR 1%")):
        ax.plot(x,[r[key] for r in A],marker="o");ax.set(xlabel="reference distance d (px)",title=title);ax.grid(alpha=.3)
    fig.savefig(o/"metric_plots"/"experiment_a_distance.png",dpi=150);plt.close(fig)
    names=[n for n,_ in exp.B_FACTORS];fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for ax,key,title in zip(axes,("roc_auc","pr_auc","defect_detection_rate_fpr1"),("ROC-AUC","PR-AUC","Defect detection @ FPR 1%")):
        for mode in exp.W_MODES:
            q=[r for r in B if r["window_mode"]==mode];q.sort(key=lambda r:names.index(r["distance_name"]));ax.plot(range(4),[r[key] for r in q],marker="o",label=mode)
        ax.set_xticks(range(4),names);ax.set_title(title);ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.savefig(o/"metric_plots"/"experiment_b_window_distance.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    labels=[];mean=[];median=[]
    for d in ("d48","d96","d128"):
        labels.append(d);mean.append(next(r["pr_auc"] for r in Cf if r["distance_name"]==d and r["aggregation"]=="mean"));median.append(next(r["pr_auc"] for r in Cf if r["distance_name"]==d and r["aggregation"]=="median"))
    xx=np.arange(3);axes[0].plot(xx,mean,marker="o",label="mean");axes[0].plot(xx,median,marker="o",label="median");axes[0].set_xticks(xx,labels);axes[0].set_title("W=128 common-region PR-AUC");axes[0].grid(alpha=.3);axes[0].legend()
    axes[1].bar([-.18,.82],[next(r["pr_auc"] for r in Cs if r["aggregation"]=="mean"),next(r["pr_auc"] for r in Cs if r["aggregation"]=="median")],width=.36,tick_label=["mean","median"]);axes[1].set_title(f"Validation: {selected_mode}/{selected_distance}\nPR-AUC (2 evaluable defects)");axes[1].grid(axis="y",alpha=.3)
    fig.savefig(o/"metric_plots"/"experiment_c_mean_median.png",dpi=150);plt.close(fig)
    # Fixed-centre contamination trend in A.
    contamA=[r for r in cont if r["experiment"]=="A" and r["target_kind"]=="fixed_center"]
    trend=[]
    for d in orderA:
        q=[r for r in contamA if r["distance_name"]==d];trend.append({"d":int(d[1:]),"score":np.mean([r["rms_score"] for r in q]),"refs":np.mean([r["contaminated_reference_count"] for r in q]),"frac":np.mean([r["reference_defect_fraction_mean"] for r in q]),"overlap":np.mean([r["target_reference_overlap_mean"] for r in q]),"n":len(q)})
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True);ax2=ax.twinx();ax.plot([r["d"] for r in trend],[r["score"] for r in trend],marker="o",color="crimson",label="RMS score");ax2.plot([r["d"] for r in trend],[r["refs"] for r in trend],marker="s",color="navy",label="contaminated refs");ax.set(xlabel="d (px)",ylabel="fixed-centre RMS (dB)");ax2.set_ylabel("mean contaminated-reference count");ax.grid(alpha=.3);fig.savefig(o/"metric_plots"/"contamination_vs_distance.png",dpi=150);plt.close(fig)
    # Japanese report.
    lines=["# 局所FFTによる傷検出：参照距離と参照集約方法", "", "## 1. 現在の実装", "",
           "処理順は、窓内平均除去 → separable 2-D Hann → `rfft2`（backward scaling）→ 線形power=|F|² → 共役重み付き放射平均 → 各窓を10log10でdB化 → 8参照dBスペクトルを周波数別に集約 → 対象との差のRMSである。総パワー正規化、ゼロパディング、格子ノッチ除去は行わない。", "",
           "基準meanは **方式B：各参照窓をdB変換してから8本を算術平均** である。方式A（線形パワーを8窓平均してからdB化）ではない。medianもdB変換後の8本を周波数ごとに中央値集約した。対象中心から上下左右・斜めへ距離dだけ移動した8窓を使い、8窓が揃わない位置は無効とした。ヒートマップは中心スコアを16×16 pxセルに戻した。", "",
           "## 2. 実験の分離", "", "- 実験A：W=128、mean、0.02–0.50 cycles/pixelを固定し、dだけを48〜128 pxに変更。", "- 実験B：meanを固定し、W=1P/2P/3Pとd=48/0.5W/0.75W/1.0Wを比較。", "- 実験C：FFT・帯域・参照位置を固定してmean/medianだけを変更。周期窓規則はimage2〜10で選び、image11〜18で独立検証。", "",
           "## 3. 実験A", "", "|d|ROC-AUC|PR-AUC|傷検出@FPR1%|@FPR5%|評価傷|", "|---:|---:|---:|---:|---:|---:|"]
    for r in A:lines.append(f"|{r['distance_name'][1:]}|{r['roc_auc']:.4f}|{r['pr_auc']:.4f}|{int(r['detected_defects_fpr1'])}|{int(r['detected_defects_fpr5'])}|{int(r['evaluable_defects'])}|")
    lines += ["", f"ROC-AUC/PR-AUCはd=64が最大。一方、固定FPRの傷検出はd=48が最多で、分離順位とLOIO絶対閾値の順位は一致しなかった。固定中心の平均RMSはd=48の{trend[0]['score']:.3f}からd=128の{trend[-1]['score']:.3f} dBへ増加したが、正常側変動も増えるためAUCは単調増加しなかった。", "",
              "## 4. 実験B", "", "|W規則|d規則|ROC-AUC|PR-AUC|傷検出@FPR1%|評価傷|", "|---|---|---:|---:|---:|---:|"]
    for mode in exp.W_MODES:
        for name,_ in exp.B_FACTORS:
            r=next(x for x in B if x["window_mode"]==mode and x["distance_name"]==name);lines.append(f"|{mode}|{name}|{r['roc_auc']:.4f}|{r['pr_auc']:.4f}|{int(r['detected_defects_fpr1'])}|{int(r['evaluable_defects'])}|")
    lines += ["", f"PR-AUC最大は **{bestB['window_mode']}/{bestB['distance_name']}** ({bestB['pr_auc']:.4f})。2P・3Pでは固定48より比例距離でROC/PRが大きく改善し、以前の大窓スコア低下には高い窓重なりと参照傷混入が寄与していた。ただし共通領域は14傷に縮小し、固定FPR検出は固定48が優位な条件もある。", "",
              "## 5. 実験C", "", "|条件|mean PR-AUC|median PR-AUC|mean/median 傷検出@FPR1%|評価傷|", "|---|---:|---:|---:|---:|"]
    for d in ("d48","d96","d128"):
        m=next(r for r in Cf if r["distance_name"]==d and r["aggregation"]=="mean");md=next(r for r in Cf if r["distance_name"]==d and r["aggregation"]=="median");lines.append(f"|W128 {d}|{m['pr_auc']:.4f}|{md['pr_auc']:.4f}|{int(m['detected_defects_fpr1'])}/{int(md['detected_defects_fpr1'])}|{int(md['evaluable_defects'])}|")
    sm=next(r for r in Cs if r["aggregation"]=="mean");smd=next(r for r in Cs if r["aggregation"]=="median");lines.append(f"|検証 {selected_mode}/{selected_distance}|{sm['pr_auc']:.4f}|{smd['pr_auc']:.4f}|{int(sm['detected_defects_fpr1'])}/{int(smd['detected_defects_fpr1'])}|{int(smd['evaluable_defects'])}|")
    lines += ["", "W=128では全3距離でmedianがROC/PRと固定FPR検出を改善した。medianは傷スコア自体を必ず増やすのではなく、正常分布と汚染参照の影響を抑えて分離を改善した。開発側で選んだ3P/1.0Wでも独立検証PR-AUCはmedianが上だったが、評価可能傷が2件しかなく一般化の根拠としては弱い。", "",
              "## 6. 傷混入分析", "", "|d|固定中心RMS|混入参照数|参照傷率|窓重なり率|n|", "|---:|---:|---:|---:|---:|---:|"]
    for r in trend:lines.append(f"|{r['d']}|{r['score']:.3f}|{r['refs']:.2f}|{r['frac']:.4f}|{r['overlap']:.4f}|{r['n']}|")
    lines += ["", "固定中心ではd増加に伴い参照混入と重なりが低下し、傷RMSは増加した。全条件を混ぜたSpearman相関は " + ", ".join(f"{r['aggregation']} ρ={r['spearman_contamination_vs_rms']:.3f} (n={int(r['n'])})" for r in cor) + "。窓サイズ・画像差を含むため因果係数ではないが、混入がRMSを抑える仮説と方向は一致する。", "",
              "## 7. 最終回答", "",
              "- **Q1:** 分離性能を主指標にするとW=128の最良は **d=64 px**。固定FPR傷検出数を優先するとd=48 px。dを大きくすれば傷RMSは増えるが、正常変動も増えるためd=128は最良ではない。",
              "- **Q2:** d=Wは2P・3PのROC/PR改善には有効だったが、1Pでは0.75Wが上で、評価領域も大きく減る。普遍的な指定にはできない。",
              "- **Q3:** 主因の一つは参照への傷混入・窓重なりである。2P/3Pは比例距離で分離が回復したため、大窓そのものだけが原因ではない。ただし大窓は傷成分の希釈と有効領域縮小を残すため、両要因が存在する。",
              "- **Q4:** **W=128ではmedianが明確に優位**。周期3P/1.0Wでも検証PR-AUCはmedianが上だが2傷のみなので追加検証が必要。",
              f"- **Q5:** 現時点の候補は、分離重視なら **W=2〜3P、d=0.75〜1.0W、median**、有効領域と固定FPR検出重視なら **W=128、d=48〜64、median**。単一ルール確定には独立画像で2P/3P候補を再検証する必要がある。",
              "- **Q6:** W=128 medianでは点状・線状・局所変形の全てでmeanより固定FPR検出が改善したが、件数は点状11、線状3、局所変形15。面状異常は0件で結論不能。詳細は `per_type_metrics.csv`。", "",
              "## 8. 評価上の注意", "", "- A/B閾値はLOIOの画像等重み正常p99/p95。Cの選択規則はimage2〜10だけで選び、image11〜18へ適用。", "- tag/backgroundは正常から除外し、傷との重複は傷優先。", "- 共通領域と全有効領域を分離。評価不能傷は失敗に含めない。", "- 傷中心固定スコアと最大スコアを `per_defect_metrics.csv` に併記。", "- image2〜18には傷36件、面状異常0件。", "", "## 9. 出力", "", "A/B/C CSV、傷別・種類別・混入分析・有効領域CSV、全画像の共有/個別正規化ヒートマップ、代表画像7/8/11/13の配置・2D/1Dスペクトル図を保存した。", "", "スペクトル詳細図の右端2列には、8-ref mean/median 2-D dBから復元した参照振幅の合成逆FFT像と、対象振幅との差の逆FFT像を追加した。log-powerは位相を持たず逆FFTが一意に定まらないため、いずれも対象FFTの位相を保持した診断像であり、検出スコアそのものの逆変換ではない。"]
    (o/"report_ja.md").write_text("\n".join(lines),encoding="utf-8")
    settings={"created_at":datetime.now().astimezone().isoformat(),"data":str(exp.DATA.resolve()),"fft_order":"subtract patch mean -> 2-D Hann -> rfft2 -> |F|^2 -> conjugate-weighted radial mean -> 10log10 per window -> aggregate 8 dB spectra -> RMS difference","reference_order":"B: aggregate after dB conversion","band_cycles_per_pixel":exp.BAND,"radial_bin_width":exp.BIN_WIDTH,"stride_px":exp.STRIDE,"experiment_A_distances_px":exp.A_DISTANCES,"experiment_B_distance_rules":[n for n,_ in exp.B_FACTORS],"development_images":exp.DEV_IMAGES,"validation_images":exp.VALIDATION_IMAGES,"selected_rule":{"window_mode":selected_mode,"distance":selected_distance},"thresholds":"A/B LOIO equal-image normal p99/p95; selected C development-only normal p99/p95","fft_score_seconds_sum":sum(r["seconds"] for r in timing),"commands":["py -3.10 reference_distance_aggregation_experiment.py --output output_reference_distance_and_aggregation --workers 6 --batch-size 64","py -3.10 reference_distance_aggregation_render.py --image N --output output_reference_distance_and_aggregation","py -3.10 finalize_reference_distance_aggregation.py --output output_reference_distance_and_aggregation"],"python":sys.version}
    (o/"settings.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding="utf-8")
    print("finalized",o)


if __name__=="__main__":main()
