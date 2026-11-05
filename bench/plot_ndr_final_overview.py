#!/usr/bin/env python3
"""Publication overview for prepared, frozen-holdout, and ordered-history data."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy import stats

def _sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def _read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def _finite(v):
 if isinstance(v,float): return math.isfinite(v)
 if isinstance(v,dict): return all(_finite(x) for x in v.values())
 if isinstance(v,list): return all(_finite(x) for x in v)
 return True
def _check(d,schema,name,require_finite=True):
 if d.get('schema')!=schema or d.get('complete') is not True or d.get('errors')!=[] or d.get('artifact_validation') is not True: raise ValueError(f'{name} failed schema/completeness/validation checks')
 if require_finite and d.get('finite_saved_fields_validation') is not True:raise ValueError(f'{name} lacks finite-field validation')
 if not _finite(d): raise ValueError(f'{name} contains nonfinite data')
def _ci(vals):
 a=np.asarray(vals,float); n=len(a); m=float(np.mean(a)); sem=float(np.std(a,ddof=1)/math.sqrt(n)); t=float(stats.t.ppf(.975,n-1)); return m,m-t*sem,m+t*sem
def _baseline(d):
 _check(d,'tdgl-ndr-interval-analysis-v1','baseline',False); ps=sorted(d['points'],key=lambda x:float(x['Je']))
 if len(ps)!=4 or len({float(x['Je']) for x in ps})!=4: raise ValueError('baseline requires four unique currents')
 seeds=[int(r['seed']) for r in ps[0]['rows']]
 if len(seeds)!=8 or len(set(seeds))!=8: raise ValueError('baseline requires eight unique seeds')
 vals=[]
 for p in ps:
  if len(p['rows'])!=8 or {int(r['seed']) for r in p['rows']}!=set(seeds): raise ValueError('baseline seed sets differ')
  vals.append([float(next(r['V'] for r in p['rows'] if int(r['seed'])==s)) for s in seeds])
 frac=[(a-b)/a for a,b in zip(vals[1],vals[2])]
 return [float(p['Je']) for p in ps],seeds,vals,frac
def _holdout(d):
 _check(d,'tdgl-ndr-holdout-analysis-v1','holdout'); rows=d.get('rows',[]); js=sorted({float(r['Je']) for r in rows}); ss=sorted({int(r['seed']) for r in rows})
 if len(rows)!=56 or len(js)!=7 or len(ss)!=8 or len({(float(r['Je']),int(r['seed'])) for r in rows})!=56: raise ValueError('holdout requires exact 7x8 matrix')
 if js!=d['protocol']['currents'] or ss!=sorted(d['protocol']['seeds']):raise ValueError('holdout declared matrix mismatch')
 if d.get('selected_candidate',{}).get('variant') is None and d.get('protocol',{}).get('selected_candidate',{}).get('variant') is None: raise ValueError('holdout selected candidate missing')
 primary=d.get('primary',{}); lo=float(primary['lo']); hi=float(primary['hi']); seedrows=primary.get('seed_rows',[]);declared=d['protocol']['primary_contrast']
 if lo!=declared['lo'] or hi!=declared['hi'] or lo not in js or hi not in js or declared['only'] is not True:raise ValueError('holdout frozen primary bracket mismatch')
 if len(seedrows)!=8 or primary.get('stats',{}).get('n')!=8 or {int(r['seed']) for r in seedrows}!=set(ss): raise ValueError('holdout primary n=8 missing')
 by={(float(r['Je']),int(r['seed'])):float(r['V']) for r in rows};saved={int(r['seed']):r for r in seedrows};fractions=[]
 for s in ss:
  a,b=by[(lo,s)],by[(hi,s)]
  if a<.05:raise ValueError('invalid primary voltage denominator')
  f=(a-b)/a
  if not math.isclose(f,float(saved[s]['fraction']),rel_tol=0,abs_tol=1e-12):raise ValueError('primary fraction disagrees with frozen endpoint data')
  fractions.append(f)
 return js,ss,rows,lo,hi,fractions
def _history(d):
 _check(d,'tdgl-full-curl-ndr-history-analysis-v1','history'); rows=d.get('rows',[]); dirs=sorted({r.get('direction') for r in rows}); js=sorted({float(r['Je']) for r in rows}); ss=sorted({int(r['seed']) for r in rows})
 if len(rows)!=56 or dirs!=['ascending','descending'] or len(js)!=7 or len(ss)!=4 or len({(r.get('direction'),float(r['Je']),int(r['seed'])) for r in rows})!=56: raise ValueError('history requires exact direction/current/seed matrix')
 if js!=d['protocol']['currents'] or ss!=sorted(d['protocol']['seeds']):raise ValueError('history declared matrix mismatch')
 return js,ss,rows
def run(b,h,hist,out,bp,hp,his):
 bx,bs,bv,bfrac=_baseline(b); hx,hs,hr,lo,hi,hfrac=_holdout(h); cx,cs,cr=_history(hist)
 if cx!=hx or not set(cs)<=set(hs) or hist['protocol']['holdout_protocol_sha256']!=h['provenance']['protocol_sha256']:raise ValueError('history is not linked to the displayed holdout')
 out.mkdir(parents=True,exist_ok=True)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(2,2,figsize=(14,10),constrained_layout=True); ax=ax.ravel()
 def prepared(axis,x,seeds,rows,key,color,label,shade=None):
  by={(float(r['Je']),int(r['seed'])):float(r[key]) for r in rows};
  for s in seeds: axis.plot(x,[by.get((j,s),np.nan) for j in x],color=color,lw=.7,alpha=.35)
  means=[]; lo_ci=[]; hi_ci=[]
  for j in x:
   v=[by[(j,s)] for s in seeds if (j,s) in by]; m,l,u=_ci(v) if len(v)>1 else (np.nan,np.nan,np.nan);means.append(m);lo_ci.append(l);hi_ci.append(u)
  axis.plot(x,means,'o-',color=color,lw=2,label=label);axis.fill_between(x,lo_ci,hi_ci,color=color,alpha=.14)
  if shade: axis.axvspan(*shade,color=color,alpha=.10)
 prepared(ax[0],bx,bs,[{'Je':x,'seed':s,'V':v} for x,vs in zip(bx,bv) for s,v in zip(bs,vs)],'V','#0072B2','Baseline prepared · 8 seeds',(bx[1],bx[2]))
 prepared(ax[1],hx,hs,hr,'V','#D55E00','Frozen candidate · 8 fresh seeds',(lo,hi))
 ax[0].text(.03,.04,f"{bx[1]:.6f} → {bx[2]:.6f}\nMean paired decrease: {100*np.mean(bfrac):.2f}%",transform=ax[0].transAxes,fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
 ax[1].text(.03,.04,f"Frozen {lo:.4f} → {hi:.4f}\nMean paired decrease: {100*np.mean(hfrac):.2f}%",transform=ax[1].transAxes,fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
 # Ordered histories: each direction and seed remains a separate trace.
 for direction,color in [('ascending','#009E73'),('descending','#7B2CBF')]:
  for s in cs:
   rr=sorted([r for r in cr if r['direction']==direction and int(r['seed'])==s],key=lambda r:float(r['Je'])); ax[2].plot([r['Je'] for r in rr],[r['V'] for r in rr],color=color,lw=.8,alpha=.35)
  means=[]; lows=[]; highs=[]
  for j in cx:
   v=[float(r['V']) for r in cr if r['direction']==direction and math.isclose(float(r['Je']),j,abs_tol=1e-12)];m,l,u=_ci(v);means.append(m);lows.append(l);highs.append(u)
  ax[2].plot(cx,means,'o-',color=color,lw=2,label=f'{direction} · 4 seeds');ax[2].fill_between(cx,lows,highs,color=color,alpha=.12)
 jitter=np.linspace(-.08,.08,8)
 ax[3].axhline(0,color='k',lw=.7);ax[3].plot(jitter,bfrac,'o',color='#0072B2',label='Baseline middle interval · 8 seeds');ax[3].plot(1+jitter,hfrac,'o',color='#D55E00',label='Fresh holdout primary · 8 seeds');ax[3].set_xticks([0,1],['Baseline','Fresh holdout']);ax[3].set_ylabel('(V_lo−V_hi)/V_lo');ax[3].set_title(f'D. Frozen primary contrast · enhancement target '+('met' if h['primary']['stronger_drop_pass'] else 'not met'));ax[3].legend(fontsize=8);ax[3].grid(alpha=.2)
 variant=h['protocol']['selected_candidate']['variant'];candidate={'T0_0p22':r'$T_0=0.22$'}.get(variant,variant)
 ax[0].set(xlabel='Je (native model units)',ylabel='V',title=r'A. Baseline $T_0=0.18$ · prepared 48–50kτ');ax[1].set(xlabel='Je (native model units)',ylabel='V',title=f'B. Candidate {candidate} · prepared 48–50kτ');ax[2].set(xlabel='Je (native model units)',ylabel='V',title=f'C. {candidate} · ordered 20kτ dwell / final 18–20kτ')
 for a in ax[:3]:a.grid(alpha=.2);a.legend(fontsize=8)
 allv=list(np.asarray(bv,float).ravel())+[float(r['V']) for r in hr]+[float(r['V']) for r in cr]
 for vals in bv:allv.extend(_ci(vals)[1:])
 for j in hx:allv.extend(_ci([float(r['V']) for r in hr if float(r['Je'])==j])[1:])
 for direction in ('ascending','descending'):
  for j in cx:allv.extend(_ci([float(r['V']) for r in cr if r['direction']==direction and float(r['Je'])==j])[1:])
 ymin,ymax=min(allv),max(allv);pad=max(.02,.06*(ymax-ymin))
 for a in ax[:3]:a.set_ylim(ymin-pad,ymax+pad)
 fig.suptitle('Full-curl I–V: prepared points and ordered histories',fontsize=15);fig.supxlabel('Bands: nominal 95% seed CI. Prepared lines guide the eye; dwell/checkpoint protocols differ.\nA stationary continuous negative branch is not confirmed.',fontsize=9.5);fig.savefig(out/'ndr_final_overview.png',dpi=180);fig.savefig(out/'ndr_final_overview.pdf');plt.close(fig)
 summary={'schema':'tdgl-ndr-final-overview-v1','input_sha256':{'baseline':_sha(bp),'holdout':_sha(hp),'history':_sha(his)},'baseline_middle_fraction':float(np.mean(bfrac)),'holdout_primary_fraction':float(np.mean(hfrac)),'holdout_stats':h.get('primary',{}).get('stats'),'enhancement_target_met':h['primary']['stronger_drop_pass'],'stationary_branch_confirmed':hist['stationary_branch_confirmed'],'code_sha256':_sha(__file__)}; (out/'overview_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--baseline',required=True,type=Path);p.add_argument('--holdout',required=True,type=Path);p.add_argument('--history',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args(argv);b=_read(a.baseline);h=_read(a.holdout);his=_read(a.history);run(b,h,his,a.out,a.baseline,a.holdout,a.history)
if __name__=='__main__':main()
