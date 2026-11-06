#!/usr/bin/env python3
"""Full-range context and eight-seed zoom of the archived full-curl voltage drop."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import t

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); study=args.root/'studies/full_curl_ndr'
    global_path=study/'full_iv/results/full_iv_analysis.json'
    local_path=study/'results/interval/analysis.json'
    full=json.loads(global_path.read_text()); local=json.loads(local_path.read_text())
    assert full['schema']=='tdgl-full-iv-analysis-v1' and full['complete'] and full['errors']==[]
    assert full['artifact_validation'] and full['finite_saved_fields_validation']
    assert local['schema']=='tdgl-ndr-interval-analysis-v1' and local['complete'] and local['errors']==[] and local['artifact_validation']
    points=sorted(full['points'],key=lambda p:p['Je'])
    assert len(points)==101 and len({p['Je'] for p in points})==101
    assert all(p['complete'] and not p['errors'] and p['seed']==2026110411 for p in points)
    x=np.array([p['Je'] for p in points]); y=np.array([p['V'] for p in points])
    focus=sorted(local['points'],key=lambda p:p['Je']); seeds=list(range(2026110411,2026110419))
    assert len(focus)==4 and all(sorted(r['seed'] for r in p['rows'])==seeds for p in focus)
    fx=np.array([p['Je'] for p in focus])
    values=np.array([[next(r['V'] for r in p['rows'] if r['seed']==s) for p in focus] for s in seeds])
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(values))
    assert np.allclose(fx,[.20816326530612245,.20918367346938774,.21020408163265306,.21122448979591835],rtol=0,atol=1e-14)
    # The global seed is already included in the local cohort, never a ninth seed.
    for i,je in enumerate(fx):
        match=np.flatnonzero(np.isclose(x,je,rtol=0,atol=1e-14))
        assert len(match)==1 and abs(y[match[0]]-values[0,i])<1e-12
    means=values.mean(axis=0); half=t.ppf(.975,7)*values.std(axis=0,ddof=1)/np.sqrt(8)
    fractions=(values[:,1]-values[:,2])/values[:,1]
    drop=float(fractions.mean()); dv=float(means[1]-means[2])
    args.out.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.size':12,'axes.titlesize':14,'axes.labelsize':12,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(13.2,5.9),layout='constrained')
        fig.suptitle('Full-curl TDGL: full I–V and the local voltage drop\n'+r'$T_0=0.18$ · final 48–50kτ of independently prepared 50kτ trajectories',fontsize=16)
        axes[0].plot(x,y,'o-',color='#0072B2',ms=3.4,lw=1.25,label='101 currents · one seed')
        axes[0].axvspan(fx[0],fx[-1],color='#D55E00',alpha=.12)
        mask=(x>=fx[1]-1e-14)&(x<=fx[2]+1e-14)
        axes[0].plot(x[mask],y[mask],'o-',color='#D55E00',lw=2.3,ms=5)
        axes[0].annotate('Local drop\nsee panel B',xy=(float(np.mean(fx[1:3])),float(means[1])),xytext=(.09,float(y.max())*.82),arrowprops=dict(arrowstyle='->',color='#333333'),fontsize=12)
        axes[0].set(title='A. Full current range',xlim=(-.004,.304),ylim=(-.1,float(y.max())*1.05))
        axes[0].legend(loc='upper left',fontsize=10)
        for i,row in enumerate(values):
            axes[1].plot(fx,row,'o-',ms=3,lw=.9,color='#7D9AB0',alpha=.55,label='Individual seeds' if i==0 else None)
        axes[1].errorbar(fx,means,yerr=half,fmt='o-',lw=2.3,ms=7,capsize=4,color='#0072B2',label='8-seed mean ± nominal 95% CI',zorder=4)
        axes[1].axvspan(fx[1],fx[2],color='#D55E00',alpha=.10)
        axes[1].plot(fx[1:3],means[1:3],'o-',color='#D55E00',lw=3,ms=7,zorder=5)
        axes[1].annotate(f'{100*drop:.2f}% paired decrease\nAll 8 seeds decrease',xy=(float(np.mean(fx[1:3])),float(np.mean(means[1:3]))),xytext=(.2100,2.85),arrowprops=dict(arrowstyle='->',color='#333333'),fontsize=12)
        low=min(float(values.min()),float((means-half).min())); high=max(float(values.max()),float((means+half).max()))
        axes[1].set(title='B. Zoom: local decrease',xlim=(float(fx.min())-.00015,float(fx.max())+.00015),ylim=(low-.035,high+.09))
        axes[1].legend(loc='upper left',fontsize=10)
        for axis in axes:
            axis.set_xlabel(r'Applied current parameter, $J_e$ (model units)');axis.set_ylabel('Voltage, V (model units)')
            axis.grid(alpha=.18);axis.ticklabel_format(axis='x',style='plain',useOffset=False)
            axis.spines[['top','right']].set_visible(False)
        fig.supxlabel('Lines guide comparisons between prepared points. A continuous stationary NDR branch remains unconfirmed.',fontsize=10)
        for ext in ('png','pdf','svg'):
            path=args.out/f'ndr_full_and_zoom.{ext}'
            fig.savefig(path,dpi=220,facecolor='white')
            if ext=='svg':
                # Matplotlib emits spaces before path-data newlines; trim for git checks.
                path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
        plt.close(fig)
    with (args.out/'ndr_zoom_source.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['Je','seed','V'])
        writer.writerows((float(je),s,float(values[i,j])) for i,s in enumerate(seeds) for j,je in enumerate(fx))
    manifest=dict(source_sha256={str(p.relative_to(args.root)):sha(p) for p in (global_path,local_path)},script_sha256=sha(Path(__file__)),
        global_points=101,global_seed=2026110411,zoom_seeds=seeds,mean_paired_decrease=drop,mean_voltage_decrease=dv,
        means=means.tolist(),nominal95_half_widths=half.tolist(),software=dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,scipy=scipy.__version__),transformations=['No smoothing; full-range point values unchanged','Zoom seed means and two-sided Student 95% intervals, df=7'],
        scope='Prepared-point voltage decrease; global seed included in zoom cohort; stationary continuous NDR unconfirmed.')
    (args.out/'figure_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'mean_paired_decrease':drop,'output':str(args.out)}))

if __name__=='__main__':main()
