"""Zoom into the low-frequency 1D FFT bins for one image and block size."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from block_1d_fft_experiment import ROOT, read_gray
from fft2_cosine_similarity import one_d_block_spectra


def run(image_name='image15.png',block_size=512,fmax=.03,output_root=ROOT/'outputs/fft2_peak_match_512',
        input_dir=ROOT/'data20260903/data20260903'):
    output_root=Path(output_root); input_dir=Path(input_dir)
    group=output_root/'results_1d_fft'/str(block_size)/Path(image_name).stem
    spectra=np.load(group/'reference_spectra.npz')
    fx=spectra['frequency_x']; fy=spectra['frequency_y']
    refx=spectra['reference_x']; refy=spectra['reference_y']
    with (output_root/'similarity_3class_blocks.csv').open(encoding='utf-8-sig',newline='') as stream:
        classified={(r['image_name'],int(r['block_size']),r['block_id']):r for r in csv.DictReader(stream)}
    with (output_root/'results_1d_fft'/'blocks_1d_fft_scores.csv').open(encoding='utf-8-sig',newline='') as stream:
        scores=[r for r in csv.DictReader(stream) if r['image_name']==image_name and int(r['block_size'])==block_size]
    score_by_id={r['block_id']:r for r in scores}
    gray=read_gray(input_dir/image_name); n=block_size
    traces=[]
    for r in classified.values():
        if r['image_name']!=image_name or int(r['block_size'])!=n: continue
        if r['similarity_class']=='background': continue
        x,y=int(r['x']),int(r['y']); block=gray[y:y+n,x:x+n]
        if block.shape!=(n,n): continue
        sx,sy=one_d_block_spectra(block,1)
        traces.append((r['block_id'],r['similarity_class'],sx,sy))
    top=sorted((r for r in scores if r['rank']),key=lambda r:int(r['rank']))[:10]
    out=output_root/'results_1d_fft'/'low_frequency_zoom'; out.mkdir(parents=True,exist_ok=True)
    def decorate(ax,direction):
        ax.set_xlim(0,fmax); ax.set_xlabel('Frequency [cycles/pixel]'); ax.set_ylabel('Mean FFT amplitude')
        ax.set_title(f'{image_name} / {n}px / {direction} FFT, low-frequency zoom'); ax.grid(alpha=.25)
        for cycles,color in ((3,'#9467bd'),(4,'#8c564b')):
            f=cycles/n
            if f<=fmax:
                ax.axvline(f,color=color,ls='--',lw=1.2,label=f'{cycles} cycles / block (bin {cycles})')
        ax.legend(fontsize=8)
    # Every eligible block is shown faintly, with the grid-only reference overlaid.
    for direction,freq,ref,index,color in [('x',fx,refx,2,'#20a45b'),('y',fy,refy,3,'#20a45b')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for block_id,category,sx,sy in traces:
            values=sx if direction=='x' else sy
            ax.plot(freq,values,color=color if category=='grid' else '#ef8b2c',alpha=.18 if category=='grid' else .42,lw=.8)
        ax.plot(freq,ref,color='#d62728',lw=2.4,label='grid reference mean')
        decorate(ax,direction.upper()); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out/f'{Path(image_name).stem}_{n}_all_blocks_{direction}_zoom.png',dpi=160); plt.close(fig)
    # Compare the ten lowest min(sim_x, sim_y) blocks directly with the reference.
    for direction,freq,ref in [('x',fx,refx),('y',fy,refy)]:
        fig,ax=plt.subplots(figsize=(10,5.5)); ax.plot(freq,ref,color='black',lw=2.5,label='grid reference mean')
        for r in top:
            found=next((t for t in traces if t[0]==r['block_id']),None)
            if found is None: continue
            values=found[2] if direction=='x' else found[3]
            ax.plot(freq,values,lw=1.25,alpha=.8,label=f"#{int(r['rank'])} {r['block_id']} ({float(r['min_similarity']):.3f})")
        decorate(ax,direction.upper()); ax.legend(fontsize=7,ncol=2)
        fig.tight_layout(); fig.savefig(out/f'{Path(image_name).stem}_{n}_top10_{direction}_zoom.png',dpi=160); plt.close(fig)
    (out/'README.md').write_text(f'# Low-frequency FFT zoom: {image_name}, {n}px\n\n'
        f'現在の処理と同じく各行/列のrFFT振幅を平均し、DC bin 0のみ除外しています。表示範囲は0〜{fmax} cycles/pixel。縦破線は512pxブロック内に3周期/4周期ある場合の周波数（bin 3/4）です。背景分類されたブロックは除外し、緑=grid、橙=uncertain、赤=grid reference mean。detrendや低周波除去は追加していません。')
    print(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',default='image15.png')
    parser.add_argument('--block-size',type=int,default=512)
    parser.add_argument('--fmax',type=float,default=.03)
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/fft2_peak_match_512')
    parser.add_argument('--input',type=Path,default=ROOT/'data20260903/data20260903')
    args=parser.parse_args()
    run(args.image,args.block_size,args.fmax,args.output_root,args.input)


if __name__=='__main__': main()
