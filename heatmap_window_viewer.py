"""Interactive viewer for strict-B8 FFT heatmaps, grid points and annotations."""
from pathlib import Path
import json, re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from matplotlib.patches import Rectangle, Polygon

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data20260903'/'data20260903'; MAPS=ROOT/'output_strict_b8_window_maps'/'maps'; WS=(32,64,128,256,512)
def shapes(i): return json.loads((DATA/f'image{i}.json').read_text(encoding='utf-8'))['shapes']
def period(i):
    q={}
    for s in shapes(i):
      m=re.match(r'grid\d+_r(\d+)-?_c(\d+)',s['label'])
      if m and s.get('points'):q[(int(m.group(1)),int(m.group(2)))]=np.mean(s['points'],axis=0)
    h=[np.linalg.norm(q[r,c+1]-q[r,c]) for r,c in q if (r,c+1) in q];v=[np.linalg.norm(q[r+1,c]-q[r,c]) for r,c in q if (r+1,c) in q]
    return (np.median(h),np.median(v),(np.median(h)*np.median(v))**.5) if h and v else (np.nan,np.nan,np.nan)
def draw(i,w):
    ax_img.clear(); ax_map.clear(); ax_img.imshow(Image.open(DATA/f'image{i}.png').convert('L'),cmap='gray')
    path=MAPS/f'image{i}_w{w}.npz'
    pu,pv,p=period(i)
    if not path.exists():
      ax_map.set_title(f'W={w}px: not evaluated (eight references do not fit)'); ax_map.set_facecolor('black'); info.set_text(f'image{i} | FFT window W={w}px | grid period p={p:.1f}px | W/p={w/p:.2f}\nNo valid locations with all 8 reference blocks'); fig.canvas.draw_idle(); return
    with np.load(path) as z: a=z['anomaly_map']; x=z['x_centers']; y=z['y_centers']
    vals=a[np.isfinite(a)]; im=ax_map.imshow(a,extent=[x[0]-8,x[-1]+8,y[-1]+8,y[0]-8],cmap='inferno',vmin=np.percentile(vals,1),vmax=np.percentile(vals,99))
    info.set_text(f'image{i} | FFT window (batch size) W={w}px | grid period p={p:.1f}px (horizontal {pu:.1f}, vertical {pv:.1f}) | W/p={w/p:.2f}\nvalid locations: {len(vals):,} | RMS median: {np.median(vals):.3f} dB | RMS p95: {np.percentile(vals,95):.3f} dB')
    for ax in (ax_img,ax_map): ax.set_xlim(0,2048);ax.set_ylim(1536,0);ax.set_xlabel('x (px)');ax.set_ylabel('y (px)')
    pts=[]
    for s in shapes(i):
      lab=s['label']; p=s.get('points',[])
      if re.match(r'grid\d+_r\d+-?_c\d+',lab): pts.append(np.mean(p,axis=0))
      elif lab=='tag' and s.get('shape_type')=='polygon':
       for ax in (ax_img,ax_map): ax.add_patch(Polygon(p,closed=True,fill=False,edgecolor='cyan',linewidth=1.3))
      elif s.get('shape_type')=='rectangle' and p:
       (x0,y0),(x1,y1)=p
       for ax in (ax_img,ax_map): ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='lime',linewidth=1.2))
    if pts:
      q=np.array(pts)
      for ax in (ax_img,ax_map):ax.scatter(q[:,0],q[:,1],s=18,c='deepskyblue',edgecolors='black',linewidths=.4)
    ax_img.set_title(f'image{i}: original | cyan=grid points, tag outline; lime=damage boxes')
    ax_map.set_title(f'W={w}px | strict 8-reference RMS | black=NaN')
    fig.canvas.draw_idle()
fig=plt.figure(figsize=(15,8)); info=fig.text(.06,.955,'',fontsize=10,va='top'); ax_img=fig.add_axes([.06,.12,.38,.78]);ax_map=fig.add_axes([.50,.12,.38,.78]);ax_i=fig.add_axes([.90,.55,.08,.35]);ax_w=fig.add_axes([.90,.18,.08,.27]);ax_i.set_title('画像選択');ax_w.set_title('FFT窓\nサイズ W')
ri=RadioButtons(ax_i,[str(i) for i in range(1,22)],active=1);rw=RadioButtons(ax_w,[str(w) for w in WS],active=2)
state={'i':2,'w':128}
def change_i(label):state['i']=int(label);draw(**state)
def change_w(label):state['w']=int(label);draw(**state)
ri.on_clicked(change_i);rw.on_clicked(change_w);draw(**state);plt.show()
