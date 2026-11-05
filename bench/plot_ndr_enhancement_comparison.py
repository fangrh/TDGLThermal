#!/usr/bin/env python3
"""Standalone comparison figure for the bounded baseline and frozen holdout."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy import stats

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def check(d,name):
 if d.get('complete') is not True or d.get('errors'): raise ValueError(f'{name} is incomplete or has errors')
 if d.get('artifact_validation') is not True: raise ValueError(f'{name} is not artifact-verified')
 if d.get('finite_validation') is False: raise ValueError(f'{name} has nonfinite observables')
def ci(values):
 v=np.asarray(values,float); n=len(v); mean=float(np.mean(v)); sem=float(np.std(v,ddof=1)/np.sqrt(n)) if n>1 else float('nan'); t=float(stats.t.ppf(.975,n-1)) if n>1 else float('nan')
 return {'n':n,'mean':mean,'ci_low':mean-t*sem,'ci_high':mean+t*sem,'tcrit':t}
def baseline_data(d):
 ps=sorted(d['points'],key=lambda p:float(p['Je']))
 if d.get('schema')!='tdgl-ndr-interval-analysis-v1':raise ValueError('baseline schema mismatch')
 if len(ps)!=4 or len({float(p['Je']) for p in ps})!=4: raise ValueError('baseline requires four unique currents')
 seedsets=[{int(r['seed']) for r in p.get('rows',[])} for p in ps]
 if any(len(s)!=8 for s in seedsets) or any(len(p.get('rows',[]))!=8 for p in ps) or any(s!=seedsets[0] for s in seedsets[1:]): raise ValueError('baseline requires identical eight-seed rows')
 xs=[float(p['Je']) for p in ps]; seeds=sorted(seedsets[0]); vals=[[next(float(r['V']) for r in p['rows'] if int(r['seed'])==s) for s in seeds] for p in ps]; means=[float(np.mean(v)) for v in vals]; cis=[ci(v) for v in vals]; pairs=[]
 if not np.isfinite(vals).all() or min(min(v) for v in vals)<=0:raise ValueError('invalid baseline voltages')
 for i,(a,b) in enumerate(zip(vals,vals[1:])):
  frac=[(x-y)/x for x,y in zip(a,b)]; pairs.append({'left':xs[i],'right':xs[i+1],'fractions':frac,**ci(frac)})
 return xs,seeds,vals,means,cis,pairs
def run(baseline,holdout,out):
 check(baseline,'baseline'); bx,bseeds,bv,bm,bc,bpairs=baseline_data(baseline)
 if holdout is not None:
  check(holdout,'holdout')
  if holdout.get('schema')!='tdgl-ndr-holdout-analysis-v1': raise ValueError('holdout schema mismatch')
  if holdout.get('finite_saved_fields_validation') is not True: raise ValueError('holdout finite-field validation missing')
  rows=holdout.get('rows',[]); hs=sorted({int(r['seed']) for r in rows}); hj=sorted({float(r['Je']) for r in rows})
  if len(rows)!=56 or len(hs)!=8 or len(hj)!=7 or len({(int(r['seed']),float(r['Je'])) for r in rows})!=56: raise ValueError('holdout requires exact 7x8 matrix')
  if holdout.get('primary',{}).get('stats',{}).get('n')!=8 or len(holdout.get('primary',{}).get('seed_rows',[]))!=8: raise ValueError('holdout primary statistics incomplete')
 out.mkdir(parents=True,exist_ok=True)
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  ncol=3 if holdout else 2; fig,ax=plt.subplots(1,ncol,figsize=(7*ncol,5),constrained_layout=True); ax=np.atleast_1d(ax)
  for i,s in enumerate(bseeds): ax[0].plot(bx,[bv[k][i] for k in range(4)],lw=.7,alpha=.35,color='#0072B2')
  ax[0].errorbar(bx,bm,yerr=[[m-c['ci_low'] for m,c in zip(bm,bc)],[c['ci_high']-m for m,c in zip(bm,bc)]],fmt='o-',color='#0072B2',capsize=4,label='Baseline mean ± nominal 95% seed CI')
  ax[0].set(xlabel='Je (native model units)',ylabel='V (native model units)',title='Baseline I–V · 8 independent seeds');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
  mid=bpairs[1]; ax[0].axvspan(mid['left'],mid['right'],alpha=.09,color='#0072B2'); ax[0].annotate(f"Paired drop = {100*mid['mean']:.1f}%\n(n=8)",((mid['left']+mid['right'])/2,(bm[1]+bm[2])/2),xytext=(30,45),textcoords='offset points',arrowprops={'arrowstyle':'->'})
  if holdout:
   rows=holdout.get('rows',[]); currents=hj; seeds=hs; by={(float(r['Je']),int(r['seed'])):r['V'] for r in rows};
   for s in seeds: ax[1].plot(currents,[by.get((j,s),np.nan) for j in currents],lw=.7,alpha=.3)
   cs=[ci([by[(j,s)] for s in seeds]) for j in currents];means=[c['mean'] for c in cs]
   ax[1].errorbar(currents,means,yerr=[c['ci_high']-c['mean'] for c in cs],fmt='o-',capsize=4,lw=2,color='#D55E00',label='Mean ± nominal 95% seed CI');ax[1].set_title('Frozen holdout · '+holdout['protocol']['selected_candidate']['variant']);ax[1].set_xlabel('Je (native model units)');ax[1].set_ylabel('V (native model units)');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
   ax[1].axvspan(holdout['primary']['lo'],holdout['primary']['hi'],color='#D55E00',alpha=.09)
   ax[1].text(.03,.04,f"Frozen contrast: {holdout['primary']['lo']:.4f} → {holdout['primary']['hi']:.4f}\nMean paired decrease: {100*holdout['primary']['stats']['mean_fraction']:.2f}%",transform=ax[1].transAxes,fontsize=8.5,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
   all_v=np.array([*sum(bv,[]),*by.values()],dtype=float)
   if not np.isfinite(all_v).all():raise ValueError('nonfinite holdout voltages')
   pad=max(.03,.08*np.ptp(all_v))
   for a in ax[:2]:a.set_ylim(float(np.min(all_v)-pad),float(np.max(all_v)+pad))
   jitter=np.linspace(-.08,.08,8)
   ax[2].axhline(0,color='k',lw=.7); ps=holdout['primary']['seed_rows']; frac=[float(x['fraction']) for x in ps]; base_frac=bpairs[1]['fractions']; ax[2].plot(jitter,base_frac,'o',color='#0072B2',label='Baseline (8 seeds)');ax[2].plot(1+jitter,frac,'o',color='#D55E00',label='Fresh holdout (8 seeds)');ax[2].set_xticks([0,1],['Baseline','Fresh holdout']);ax[2].set_ylabel('(V_lo−V_hi)/V_lo');ax[2].set_title('Primary paired fraction distributions');ax[2].legend(fontsize=8);ax[2].grid(alpha=.2)
  else:
   ax[1].axhline(0,color='k',lw=.7); ax[1].plot(np.linspace(-.08,.08,8),bpairs[1]['fractions'],'o',color='#0072B2'); ax[1].set_xticks([0],['Baseline middle interval']); ax[1].set_ylabel('(V_lo−V_hi)/V_lo'); ax[1].set_title('Baseline paired fractions (8 seeds)'); ax[1].grid(alpha=.2)
  fig.suptitle('Full-curl I–V · independent hot starts · final 48–50kτ\nPrepared-point voltage drop; stationary negative branch unconfirmed',fontsize=12); fig.savefig(out/'ndr_enhancement_comparison.png',dpi=180);fig.savefig(out/'ndr_enhancement_comparison.pdf');plt.close(fig)
 except ImportError: raise
 summary={'schema':'tdgl-ndr-enhancement-comparison-v1','baseline_middle_drop_fraction':mid['mean'],'baseline_pairs':bpairs,'holdout_included':holdout is not None,'input_hashes':{'baseline':baseline.get('_input_sha256'),'holdout':holdout.get('_input_sha256') if holdout else None},'code_sha256':sha(__file__)}
 if holdout: summary['holdout_primary']=holdout.get('primary',{}).get('stats')
 (out/'comparison_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
 return summary
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--baseline',required=True,type=Path);p.add_argument('--holdout',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args(argv);b=read(a.baseline);b['_input_sha256']=sha(a.baseline);h=read(a.holdout) if a.holdout else None
 if h is not None:h['_input_sha256']=sha(a.holdout)
 run(b,h,a.out)
if __name__=='__main__':main()
