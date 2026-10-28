"""Draw ranked 1D-FFT blocks on their source image."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from block_1d_fft_experiment import ROOT, read_gray


def plot(image_name='image15.png',top_k=10,output_root=ROOT/'outputs/fft2_peak_match',input_dir=ROOT/'data20260903/data20260903'):
    output_root=Path(output_root); input_dir=Path(input_dir)
    source=output_root/'results_1d_fft'/'blocks_1d_fft_scores.csv'
    with source.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    gray=read_gray(input_dir/image_name)
    for size in (512,256,128):
        selected=sorted((r for r in rows if r['image_name']==image_name and int(r['block_size'])==size
                         and r['rank'] and int(r['rank'])<=top_k),key=lambda r:int(r['rank']))
        fig=plt.figure(figsize=(15,9)); layout=fig.add_gridspec(1,2,width_ratios=(4,1)); ax=fig.add_subplot(layout[0,0]); side=fig.add_subplot(layout[0,1]); ax.imshow(gray,cmap='gray')
        for r in selected:
            x,y=int(r['x0']),int(r['y0']); rank=int(r['rank'])
            ax.add_patch(Rectangle((x,y),size,size,facecolor=(1,.05,.05,.27),edgecolor='#ff2020',linewidth=2.2))
            ax.text(x+size/2,y+size/2,str(rank),ha='center',va='center',color='white',fontsize=11,weight='bold',
                    bbox=dict(boxstyle='circle,pad=.35',facecolor='#d00000',edgecolor='white',linewidth=1.2,alpha=.95))
        side.axis('off'); side.set_title('Rank / block / min score',fontsize=10)
        side.text(0,.98,'\n'.join(f"#{int(r['rank']):02d}  {r['block_id']}\n      {float(r['min_similarity']):.3f}" for r in selected),
                  va='top',ha='left',fontsize=9,family='monospace',linespacing=1.35)
        ax.set_title(f'{image_name} | {size}px | 1D FFT suspicious TOP {len(selected)} (red)\nRanked by min(sim_x, sim_y), ascending')
        ax.axis('off'); fig.tight_layout()
        dest=output_root/'results_1d_fft'/str(size)/Path(image_name).stem/'suspicious_top10_locations.png'
        dest.parent.mkdir(parents=True,exist_ok=True); fig.savefig(dest,dpi=170,bbox_inches='tight'); plt.close(fig)
        print(dest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',default='image15.png')
    parser.add_argument('--top-k',type=int,default=10)
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/fft2_peak_match')
    parser.add_argument('--input',type=Path,default=ROOT/'data20260903/data20260903')
    args=parser.parse_args()
    if args.top_k<1: parser.error('--top-k must be >=1')
    plot(args.image,args.top_k,args.output_root,args.input)


if __name__=='__main__': main()
