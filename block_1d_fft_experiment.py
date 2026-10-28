"""Projection-profile 1D FFT experiment. Run: python block_1d_fft_experiment.py
Dependencies: numpy, opencv-python, matplotlib. No resize; incomplete edges omitted.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
# Zero-based row/column. Visual candidates, not confirmed ground truth.
REPRESENTATIVES = [
    ('image7', 2, 10, 'Grid candidate'),
    ('image7', 1, 3, 'Dark background'),
    ('image7', 3, 3, 'Background / tape boundary'),
    ('image15', 4, 6, 'Grid near defect candidate'),
    ('image15', 4, 8, 'Scratch / foreign-object candidate'),
    ('image15', 3, 13, 'Tape background'),
]


def read_gray(path):
    a = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8),
                     cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if a is None or not np.isfinite(a).all():
        raise ValueError('Cannot read finite grayscale image')
    return a


def spectrum(profile, hann):
    n = len(profile)
    w = np.hanning(n) if hann else np.ones(n)
    # Remove constant brightness; Hann reduces discontinuity at profile ends.
    f = np.fft.rfft((profile - profile.mean()) * w)
    multiplicity = np.full(len(f), 2.)
    multiplicity[0] = 1.
    if n % 2 == 0:
        multiplicity[-1] = 1.  # Nyquist has no distinct negative counterpart.
    amplitude = np.abs(f) * multiplicity / w.sum()
    power = np.abs(f)**2 * multiplicity / (n * np.sum(w*w))
    # DC may return after windowing: explicitly exclude bin 0, not low bins.
    return amplitude[1:], power[1:]


def features(profile, amp, power, freq, band):
    total = float(power.sum())
    valid = total > 1e-16
    k = int(np.argmax(amp))
    near = np.abs(np.arange(len(freq)) - k) <= 1
    selected = (freq >= band[0]) & (freq <= band[1])
    band_indices = np.flatnonzero(selected)
    bk = int(band_indices[np.argmax(amp[selected])]) if band_indices.size else None
    # Ratios use linear power, never log amplitude. No defect score is defined.
    return dict(profile_mean=float(profile.mean()), profile_std=float(profile.std()),
        peak_amplitude=float(amp[k]), peak_frequency=float(freq[k]) if valid else None,
        peak_period_pixels=float(1/freq[k]) if valid else None,
        peak_amplitude_fraction=float(amp[k]/amp.sum()) if valid else None,
        peak_energy_ratio=float(power[near].sum()/total) if valid else None,
        band_amplitude=float(amp[selected].sum()), band_energy=float(power[selected].sum()),
        band_peak_amplitude=float(amp[bk]) if bk is not None else None,
        band_peak_frequency=float(freq[bk]) if bk is not None and valid else None,
        band_energy_ratio=float(power[selected].sum()/total) if valid else None,
        total_energy=total, spectrum_defined=valid,
        low_frequency_energy_ratio=float(power[freq <= 2/len(profile)].sum()/total) if valid else None)


def analyze(block, band):
    b = block.astype(np.float64)
    freq = np.fft.rfftfreq(b.shape[0])[1:]
    data = dict(block=block, frequency=freq)
    values = dict(block_mean=float(b.mean()), block_std=float(b.std()))
    for direction, profile in [('x', b.mean(axis=0)), ('y', b.mean(axis=1))]:
        data[direction+'_profile'] = profile
        for window in ['plain', 'hann']:
            amp, power = spectrum(profile, window == 'hann')
            data[f'{direction}_{window}_amplitude'] = amp
            data[f'{direction}_{window}_power'] = power
            for name, value in features(profile, amp, power, freq, band).items():
                values[f'{direction}_{window}_{name}'] = value
    return values, data


def comparison(samples, path, fmax=.5):
    fig, axes = plt.subplots(len(samples), 5, figsize=(19, 3.1*len(samples)), squeeze=False)
    peakmax = max(float(s[d+'_'+w+'_amplitude'].max()) for _,s in samples
                  for d in ['x','y'] for w in ['plain','hann'])
    for axs, (label, s) in zip(axes, samples):
        axs[0].imshow(s['block'], cmap='gray', vmin=0, vmax=255 if s['block'].dtype==np.uint8 else None)
        axs[0].set_title(label, fontsize=9)
        for direction, pa, fa in [('x', axs[1], axs[2]), ('y', axs[3], axs[4])]:
            pa.plot(s[direction+'_profile'], lw=1)
            pa.set(title=direction.upper()+' brightness profile', xlabel='Pixel', ylabel='Gray level')
            if s['block'].dtype == np.uint8:
                pa.set_ylim(0,255)
            for window in ['plain','hann']:
                fa.plot(s['frequency'],s[f'{direction}_{window}_amplitude'],label=window,lw=1)
            fa.set(title=direction.upper()+' FFT (DC excluded)', xlabel='Frequency [cycles/pixel]',
                   ylabel='Amplitude [gray level]', xlim=(0,fmax), ylim=(0,max(peakmax*1.08,1)))
            fa.legend(fontsize=7)
            fa.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path,dpi=120)
    plt.close(fig)


def save_atlas(records, out):
    # All blocks get X/Y spectra. Each page shares one amplitude scale.
    for start in range(0,len(records),48):
        page = records[start:start+48]
        fig, axes = plt.subplots(8,6,figsize=(18,19),squeeze=False)
        limit = max(float(s[d+'_hann_amplitude'].max()) for _,s in records for d in ['x','y'])
        for ax in axes.flat:
            ax.set_visible(False)
        for ax,(label,s) in zip(axes.flat,page):
            ax.set_visible(True)
            for d in ['x','y']:
                ax.plot(s['frequency'],s[d+'_hann_amplitude'],lw=.65,label=d)
            ax.set(title=label,xlim=(0,.5),ylim=(0,max(limit*1.05,1)))
            ax.tick_params(labelsize=6)
        axes[0,0].legend(fontsize=7)
        fig.suptitle('All blocks: Hann FFT amplitude; frequency [cycles/pixel]; DC excluded')
        fig.tight_layout(rect=(0,0,1,.98))
        fig.savefig(out/f'spectra_atlas_{start//48+1:02d}.png',dpi=95)
        plt.close(fig)


def write_csv(path, rows):
    if rows:
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data20260903/data20260903')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/block_1d_fft')
    p.add_argument('--block-size',type=int,default=128)
    p.add_argument('--stride',type=int,default=None)
    p.add_argument('--band',type=float,nargs=2,default=[.01,.04],metavar=('LOW','HIGH'))
    p.add_argument('--images',nargs='*',help='Optional stems, e.g. image7 image15')
    p.add_argument('--representatives',type=Path,help='JSON list of [stem,row,col,label]; zero based')
    p.add_argument('--all-plots',action='store_true',help='Also save five-panel detail for every block')
    p.add_argument('--no-atlas',action='store_true')
    args=p.parse_args()
    n=args.block_size; stride=n if args.stride is None else args.stride
    if n<8 or stride<1 or not (0<args.band[0]<args.band[1]<=.5):
        p.error('Require block-size >=8, stride >=1 and 0 < band low < high <=0.5')
    reps=json.loads(args.representatives.read_text(encoding='utf-8')) if args.representatives else (REPRESENTATIVES if n==128 and stride==128 else [])
    args.output.mkdir(parents=True,exist_ok=True)
    files=sorted(f for f in args.input.iterdir() if f.suffix.lower() in EXTENSIONS and not f.name.startswith('._')
                 and (not args.images or f.stem in args.images))
    if not files:
        p.error('No matching input images')
    allrows=[]; rep_samples=[]; rep_rows=[]; errors=[]; summaries=[]
    for path in files:
        try:
            gray=read_gray(path); h,w=gray.shape
            out=args.output/path.stem; out.mkdir(exist_ok=True)
            rows=[]; records=[]; positions=[]
            fig,ax=plt.subplots(figsize=(12,9)); ax.imshow(gray,cmap='gray')
            for row,y in enumerate(range(0,h-n+1,stride)):
                for col,x in enumerate(range(0,w-n+1,stride)):
                    values,s=analyze(gray[y:y+n,x:x+n],args.band)
                    label=next((r[3] for r in reps if r[:3]==[path.stem,row,col] or tuple(r[:3])==(path.stem,row,col)),'')
                    record=dict(filename=path.name,block_row=row,block_col=col,x_start=x,y_start=y,
                        block_size=n,stride=stride,label=label,band_low=args.band[0],band_high=args.band[1],**values)
                    rows.append(record); records.append((f'r{row} c{col}',s)); positions.append((row,col,x,y))
                    ax.add_patch(Rectangle((x-.5,y-.5),n,n,fill=False,edgecolor='cyan',lw=.35))
                    ax.text(x+n/2,y+n/2,f'{row},{col}',ha='center',va='center',fontsize=4,color='yellow')
                    if label:
                        rep_samples.append((f'{path.stem} r{row} c{col}\n{label}',s)); rep_rows.append(record)
                        ax.add_patch(Rectangle((x-.5,y-.5),n,n,fill=False,edgecolor='red',lw=2))
                    if args.all_plots or label:
                        comparison([(f'{path.stem} r{row} c{col}\n{label}',s)],out/f'block_r{row:02d}_c{col:02d}.png')
            ax.set_title(f'{path.name}: {n} x {n}, stride={stride}; labels=row,col (zero based)')
            fig.tight_layout(); fig.savefig(out/'block_locations.png',dpi=150); plt.close(fig)
            if not rows:
                raise ValueError('Image smaller than one full block')
            write_csv(out/'features.csv',rows)
            # Save every profile/spectrum for later comparison without another image read.
            arrays={key:np.stack([s[key] for _,s in records]) for key in records[0][1] if key not in ['block','frequency']}
            np.savez_compressed(out/'profiles_spectra.npz',frequency=records[0][1]['frequency'],
                                positions=np.array(positions),**arrays)
            if not args.no_atlas:
                save_atlas(records,out)
            allrows.extend(rows)
            summaries.append(dict(filename=path.name,width=w,height=h,blocks=len(rows),
                                  right_remainder=(w-n)%stride,bottom_remainder=(h-n)%stride))
            print(f'{path.name}: {len(rows)} blocks',flush=True)
        except Exception as e:
            errors.append(dict(filename=path.name,error=str(e)))
            plt.close('all'); print(f'ERROR {path.name}: {e}',flush=True)
    write_csv(args.output/'all_features.csv',allrows)
    write_csv(args.output/'representative_features.csv',rep_rows)
    if rep_samples:
        comparison(rep_samples,args.output/'representative_comparison.png')
        comparison(rep_samples,args.output/'representative_low_frequency.png',fmax=.12)
    (args.output/'run_summary.json').write_text(json.dumps(dict(block_size=n,stride=stride,band=args.band,
        representatives=reps,images=summaries,errors=errors,total_blocks=len(allrows)),indent=2),encoding='utf-8')
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    main()
