"""Plot normal and deformation-center score distributions for selected images."""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluate_all_radial_fft as prev
import multiscale_fft_experiment as base
import stride_grid_overlap_experiment as old

prev.N=256; root=Path('output_frequency_band_analysis')
for mode in ('fixed_references','band_specific_references'):
    for i in (5,15,17):
        image=np.asarray(Image.open(base.DATA/f'image{i}.png').convert('L')); labels=prev.annotation(base.DATA/f'image{i}.json',image.shape); r=old.load_result(Path('output_all_images_radial_fft')/f'image{i}'/'result.npz'); normal,entries=prev.classification(labels,r); defect=next(e['center'] for e in entries if e['label']=='局所変形'); path=root/mode/'images'/f'image{i}'
        with np.load(path/'scores_and_selections.npz') as a:
            fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
            for ax,band in zip(axes.flat,'ABCDEF'):
                score=a[f'score_{band}']; n=score[normal&np.isfinite(score)]; d=score[defect&np.isfinite(score)]; ax.hist(n,bins=45,density=True,alpha=.5,label='normal'); ax.hist(d,bins=25,density=True,alpha=.5,label='deformation center'); ax.set(title=band,xlabel='RMS dB',ylabel='density'); ax.legend(fontsize=7)
            fig.savefig(path/'normal_defect_distributions.png',dpi=120); plt.close(fig)
