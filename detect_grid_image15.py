"""Reproducible grid-detection experiment with per-step scientific figures.

The material ROI is specified manually for this experiment. Grid geometry and
phase are estimated from the pixels; no defect/normal ground truth is inferred.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates
from scipy.optimize import least_squares, root

BASE = Path(__file__).resolve().parent
OUT = BASE/'outputs/grid_detection_image15'
SOURCE = BASE/'data20260903/data20260903/image15.png'
ROI = (100, 120, 1000, 1280)  # x,y,width,height: avoid paper and material boundary.
PATCH = 48
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
NAMES = ['黒の交点', '白の中央', '黒1本・方向1', '黒1本・方向2']
GROUPS = ['black_cross', 'white_center', 'black_single', 'black_single']


def save(fig, name):
    fig.savefig(OUT/name, dpi=155, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def sample(field, xy):
    return float(map_coordinates(field, [[xy[1]], [xy[0]]], order=1, mode='nearest')[0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'Yu Gothic', 'font.size':11, 'axes.unicode_minus':False})
    image = np.asarray(Image.open(SOURCE).convert('L'), dtype=float)
    x0,y0,w,h = ROI
    raw = image[y0:y0+h, x0:x0+w]
    yy,xx = np.indices(raw.shape)
    smooth = gaussian_filter(raw, 4)
    trend = gaussian_filter(smooth, 90)
    detection = smooth-trend
    contrast = float(np.percentile(abs(detection), 99))

    fig,ax = plt.subplots(figsize=(10,8), layout='constrained')
    ax.imshow(image,cmap='gray',vmin=0,vmax=255)
    ax.add_patch(Rectangle((x0-.5,y0-.5),w,h,fill=False,ec=COLORS[0],lw=2))
    ax.set(title='01 検出範囲：紙・材料の端を避けて手動指定',xlabel='x [pixel]',ylabel='y [pixel]')
    save(fig,'01_detection_roi.png')

    fig,axes=plt.subplots(1,3,figsize=(13,6),layout='constrained')
    for ax,a,title in zip(axes,[raw,smooth,detection],['切り出した元画像','細かな反射点を抑制（σ=4 pixel）','緩やかな明るさむらを除去']):
        if a is detection: im=ax.imshow(a,cmap='gray',vmin=-contrast,vmax=contrast)
        else: im=ax.imshow(a,cmap='gray',vmin=0,vmax=255)
        ax.set_title(title);ax.set_axis_off()
    fig.suptitle('02 格子の位置を検出するための前処理（解析用の元画像は保持）')
    save(fig,'02_preprocessing.png')

    window=np.outer(np.hanning(h),np.hanning(w))
    spectrum=np.fft.fftshift(np.fft.fft2(detection*window))
    power=abs(spectrum)**2
    fx=np.fft.fftshift(np.fft.fftfreq(w));fy=np.fft.fftshift(np.fft.fftfreq(h))
    fxx,fyy=np.meshgrid(fx,fy);rad=np.hypot(fxx,fyy)
    eligible=(power==maximum_filter(power,5))&(rad>.003)&(rad<.035)&(fyy>0)
    peaks=sorted(np.argwhere(eligible),key=lambda a:power[tuple(a)],reverse=True)
    first=peaks[0];vectors=[np.array([fx[first[1]],fy[first[0]]])]
    for p in peaks[1:]:
        v=np.array([fx[p[1]],fy[p[0]]])
        if abs(np.linalg.det(np.stack([vectors[0],v])))/(np.linalg.norm(v)*np.linalg.norm(vectors[0])) > .6:
            vectors.append(v);break
    if len(vectors)!=2: raise RuntimeError('Two independent grid directions were not found.')
    initial=np.array(vectors)
    print('FFT seed vectors:',initial.tolist(),flush=True)

    fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
    logp=10*np.log10(power/power.max()+1e-12)
    im=ax.imshow(logp,origin='lower',extent=[fx[0]-.5/w,fx[-1]+.5/w,fy[0]-.5/h,fy[-1]+.5/h],
                 cmap='magma',vmin=-50,vmax=0,interpolation='nearest')
    for i,v in enumerate(initial):
        for sign in [-1,1]:
            ax.scatter(*(v*sign),facecolors='none',edgecolors=COLORS[i],s=160,lw=2)
        ax.annotate(f'方向{i+1}',v,xytext=(v[0]+.003,v[1]+.003),color=COLORS[i],arrowprops=dict(arrowstyle='->',color=COLORS[i]))
    ax.set(xlim=(-.022,.022),ylim=(-.022,.022),xlabel='横方向周波数 [cycles/pixel]',ylabel='縦方向周波数 [cycles/pixel]',title='03 2方向の周期成分を検出（±のピークを対で表示）')
    fig.colorbar(im,ax=ax,label='最大値基準のFFTパワー [dB]')
    save(fig,'03_fft_peaks.png')

    # Refine global frequency/phase jointly with a robust sinusoidal model.
    sx=xx[::8,::8].ravel()/w-.5;sy=yy[::8,::8].ravel()/h-.5
    target=detection[::8,::8].ravel()
    q=(initial*np.array([w,h])).ravel()
    def design(q):
        t1=2*np.pi*(q[0]*sx+q[1]*sy);t2=2*np.pi*(q[2]*sx+q[3]*sy)
        return np.column_stack([np.cos(t1),np.sin(t1),np.cos(t2),np.sin(t2),np.ones_like(sx)])
    coeff=np.linalg.lstsq(design(q),target,rcond=None)[0]
    p0=np.r_[q,coeff]
    lower=np.r_[q-.95,np.full(5,-np.inf)];upper=np.r_[q+.95,np.full(5,np.inf)]
    fit=least_squares(lambda p:design(p[:4])@p[4:]-target,p0,bounds=(lower,upper),loss='soft_l1',f_scale=8,max_nfev=90)
    q=fit.x[:4];coeff=fit.x[4:]
    refined=q.reshape(2,2)/[w,h]
    theta=[2*np.pi*(v[0]*xx+v[1]*yy) for v in refined]
    global_phase=[theta[i]-np.pi*(q[2*i]+q[2*i+1])-np.arctan2(coeff[2*i+1],coeff[2*i]) for i in range(2)]
    global_model=sum(np.hypot(coeff[2*i],coeff[2*i+1])*np.cos(global_phase[i]) for i in range(2))+coeff[4]

    # Local phase from complex demodulation: helps slowly varying spacing/tilt.
    phases=[];amplitudes=[];components=[]
    for carrier in theta:
        demod=detection*np.exp(-1j*carrier)
        env=gaussian_filter(demod.real,65)+1j*gaussian_filter(demod.imag,65)
        envelope_phase=np.unwrap(np.unwrap(np.angle(env),axis=1),axis=0)
        phase=carrier+envelope_phase
        phases.append(phase);amplitudes.append(2*abs(env))
        components.append(2*np.real(env*np.exp(1j*carrier)))
    model=components[0]+components[1]
    margin=80
    inside=np.s_[margin:-margin,margin:-margin]
    r_global=float(np.corrcoef(detection[inside].ravel(),global_model[inside].ravel())[0,1])
    r_local=float(np.corrcoef(detection[inside].ravel(),model[inside].ravel())[0,1])
    print('Global/local fit correlations:',r_global,r_local,flush=True)

    fig,axes=plt.subplots(1,3,figsize=(13,6),layout='constrained')
    for ax,a,title in zip(axes,[detection,global_model,model],['前処理後の実画像',f'一定間隔のモデル（相関 {r_global:.3f}）',f'局所的に位置を調整（相関 {r_local:.3f}）']):
        ax.imshow(a,cmap='gray',vmin=-contrast,vmax=contrast);ax.set_title(title);ax.set_axis_off()
    fig.suptitle('04 周期と位相を合わせて格子模様を再構成（相関は傷検出精度ではない）')
    save(fig,'04_grid_model_fit.png')

    fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained')
    for ax,fields,title in zip(axes,[global_phase,phases],['一定間隔の格子線','局所調整した格子線']):
        ax.imshow(raw,cmap='gray',vmin=0,vmax=255)
        for i,phase in enumerate(fields):
            levels=np.arange(np.ceil((phase.min()-np.pi)/(2*np.pi)),np.floor((phase.max()-np.pi)/(2*np.pi))+1)*2*np.pi+np.pi
            ax.contour(phase,levels=levels,colors=[COLORS[i]],linewidths=.8,linestyles=['solid','dashed'][i])
        ax.set(title=title,xlabel='検出範囲内 x [pixel]',ylabel='検出範囲内 y [pixel]')
    fig.suptitle('05 黒い線の中心を元画像に重ねて位置を確認')
    save(fig,'05_grid_lines_overlay.png')

    matrix=2*np.pi*refined
    envelope_offsets=np.array([np.median(phase-carrier) for phase,carrier in zip(phases,theta)])
    offsets=[(np.pi,np.pi),(0,0),(np.pi,0),(0,np.pi)]
    thresholds=[float(np.percentile(a[inside],10)*.55) for a in amplitudes]
    centers=[]
    for category,offset in enumerate(offsets):
        kranges=[range(int(np.floor((p.min()-off)/(2*np.pi)))-1,int(np.ceil((p.max()-off)/(2*np.pi)))+2) for p,off in zip(phases,offset)]
        for k in kranges[0]:
            for l in kranges[1]:
                wanted=2*np.pi*np.array([k,l])+offset
                seed=np.linalg.solve(matrix,wanted-envelope_offsets)
                if not(-30<seed[0]<w+30 and -30<seed[1]<h+30):continue
                sol=root(lambda xy:np.array([sample(p,xy) for p in phases])-wanted,seed)
                cx,cy=sol.x
                if not sol.success or not(margin<cx<w-margin and margin<cy<h-margin):continue
                phase_error=np.linalg.norm([sample(p,sol.x)-v for p,v in zip(phases,wanted)])
                if phase_error>1e-3:continue
                if any(sample(a,sol.x)<t for a,t in zip(amplitudes,thresholds)):continue
                ix=int(round(cx));iy=int(round(cy));half=PATCH//2
                patch=raw[iy-half:iy+half,ix-half:ix+half]
                assert patch.shape==(PATCH,PATCH)
                centers.append(dict(category=category,label=NAMES[category],center_x=x0+ix,center_y=y0+iy,
                                    crop_x=x0+ix-half,crop_y=y0+iy-half,size=PATCH,
                                    mean_gray=float(patch.mean()),phase_error=float(phase_error)))
    print('Category counts:',[sum(p['category']==i for p in centers) for i in range(4)],flush=True)

    fig,ax=plt.subplots(figsize=(10,10),layout='constrained')
    ax.imshow(image,cmap='gray',vmin=0,vmax=255)
    ax.add_patch(Rectangle((x0,y0),w,h,fill=False,ec='0.8',lw=1))
    for i in range(4):
        pts=[p for p in centers if p['category']==i]
        style={'facecolors':'none','edgecolors':COLORS[i]} if i in [1,3] else {'color':COLORS[i]}
        ax.scatter([p['center_x'] for p in pts],[p['center_y'] for p in pts],s=42,marker=['x','o','+','s'][i],
                   label=f'{NAMES[i]}：{len(pts)}か所',linewidths=1.2,**style)
    ax.set(xlim=(x0-30,x0+w+30),ylim=(y0+h+30,y0-30),xlabel='元画像 x [pixel]',ylabel='元画像 y [pixel]',title='06 格子内の位置で分類した切り出し中心（傷の有無は未判定）')
    ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    save(fig,'06_classified_centers.png')

    # Save raw, lossless crops; retain both orientations of single-black lines.
    for index,p in enumerate(centers):
        directory=OUT/'crops'/GROUPS[p['category']];directory.mkdir(parents=True,exist_ok=True)
        filename=f"{index:03d}_type{p['category']}_x{p['center_x']}_y{p['center_y']}.png"
        patch=image[p['crop_y']:p['crop_y']+PATCH,p['crop_x']:p['crop_x']+PATCH].astype(np.uint8)
        Image.fromarray(patch).save(directory/filename)
        p['file']=str((directory/filename).relative_to(OUT))
    examples=[]
    for i in range(4):
        pts=[p for p in centers if p['category']==i]
        if not pts:raise RuntimeError(f'No centers for category {i}')
        target=np.array([650,680])
        examples.append(min(pts,key=lambda p:np.linalg.norm(np.array([p['center_x'],p['center_y']])-target)))
    fig=plt.figure(figsize=(13,8),layout='constrained');gs=fig.add_gridspec(2,3,width_ratios=[1.4,1,1])
    ax=fig.add_subplot(gs[:,0]);ax.imshow(image,cmap='gray',vmin=0,vmax=255)
    for i,p in enumerate(examples):
        ax.add_patch(Rectangle((p['crop_x']-.5,p['crop_y']-.5),PATCH,PATCH,fill=False,ec=COLORS[i],lw=2))
        ax.text(p['crop_x'],p['crop_y']-12,str(i+1),color=COLORS[i],fontsize=14)
        a=fig.add_subplot(gs[i//2,i%2+1]);patch=np.asarray(Image.open(OUT/p['file']))
        a.imshow(patch,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        a.set_title(f"{i+1} {NAMES[i]}\n中心 ({p['center_x']}, {p['center_y']}) / 平均 {p['mean_gray']:.1f}")
        a.set_xticks([]);a.set_yticks([])
    ax.set(xlim=(300,1000),ylim=(1050,350),title='代表例の位置（元画像座標）',xlabel='x [pixel]',ylabel='y [pixel]')
    fig.suptitle(f'07 元画像から {PATCH} × {PATCH} pixel で切り出し（明るさの表示範囲は共通）')
    save(fig,'07_example_crops.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    residual_image=axes[0].imshow(detection-model,cmap='coolwarm',vmin=-contrast,vmax=contrast)
    fig.colorbar(residual_image,ax=axes[0],shrink=.8,label='前処理画像 − 格子モデル [輝度値]')
    axes[0].set(title='格子モデルとの差（傷以外の差も含む）');axes[0].set_axis_off()
    vals=[[p['mean_gray'] for p in centers if p['category']==i] for i in range(4)]
    axes[1].boxplot(vals)
    axes[1].set_xticks([1,2,3,4],['黒交点','白中央','黒1本①','黒1本②'])
    axes[1].set(ylabel='切り出し画像の平均輝度 [0–255]',title='分類した位置の明るさ分布')
    fig.suptitle('08 検出の確認：分類の整合性とモデルに合わない場所')
    save(fig,'08_validation.png')

    metadata={'source':str(SOURCE),'roi_xywh':ROI,'roi_selection':'manual; excludes paper and sample boundary',
              'crop_size':PATCH,'gaussian_sigma':4,'background_sigma':90,'demodulation_sigma':65,
              'fft_seed_vectors_cycles_per_pixel':initial.tolist(),'refined_vectors_cycles_per_pixel':refined.tolist(),
              'normal_to_stripe_periods_pixels':(1/np.linalg.norm(refined,axis=1)).tolist(),
              'global_fit_success':bool(fit.success),'global_correlation':r_global,'local_correlation':r_local,
              'correlation_note':'in-sample grid reconstruction agreement, not independent validation or defect accuracy',
              'categories':NAMES,'counts':[len(v) for v in vals],'median_crop_gray':[float(np.median(v)) for v in vals],
              'limitations':['single manually chosen ROI','grid model assumes two periodic directions','local phase can be biased by large defects','no defect/normal labels inferred'],
              'representative_crops':examples,'regions':centers}
    (OUT/'detection_results.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:metadata[k] for k in ['normal_to_stripe_periods_pixels','counts','median_crop_gray','global_fit_success']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
