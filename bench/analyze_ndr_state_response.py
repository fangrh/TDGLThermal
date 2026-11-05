#!/usr/bin/env python3
"""Receipt-verified local current-step responses; reused seeds are exploratory."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import scipy
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent))
import prepare_ndr_state_response as prep
from analyze_full_curl_ndr import read_segment
from analyze_full_iv import _finite_fields_errors
from compare_results import validate_artifact

MODEL_HASH, RHS_HASH, CONFIG_HASH = prep.MODEL_HASH, prep.RHS_HASH, prep.CONFIG_HASH
SEEDS, SOURCE, ARMS = prep.SEEDS, prep.SOURCE, ('minus', 'sham', 'plus')
DATASETS = ('times', 'V', 'T_avg', 'psi_avg', 'psi_real', 'psi_imag', 'T', 'Ax', 'Ay')

def _sha(path): return prep._sha(Path(path))
def _read(path): return prep._read(Path(path))
def _finite(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: _finite(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_finite(v) for v in value]
    return value

def close(value, expected, tol=1e-12):
    try:
        x=float(value)
        return math.isfinite(x) and math.isclose(x,expected,rel_tol=0,abs_tol=tol)
    except (ValueError,TypeError): return False

def paired(values):
    n=len(values); mean=float(np.mean(values)) if n else math.nan
    sem=float(np.std(values,ddof=1)/math.sqrt(n)) if n>1 else math.nan
    crit=float(stats.t.ppf(1-.05/4,n-1)) if n>1 else math.nan
    return dict(n=n,values=values,mean=mean,sem=sem,tcrit=crit,upper95_bonferroni4=mean+crit*sem)

def drift_stats(values):
    n=len(values); mean=float(np.mean(values)) if n else math.nan
    half=float(stats.t.ppf(.975,n-1)*np.std(values,ddof=1)/math.sqrt(n)) if n>1 else math.nan
    return dict(n=n,mean=mean,ci_low=mean-half,ci_high=mean+half,bound=.0025,
                pass_drift=bool(n==4 and abs(mean)+half<=.0025))

def dataset_digests(path):
    """Hash numeric bytes, shapes and dtypes; container metadata can differ."""
    result={}
    with h5py.File(path,'r') as h:
        for key in DATASETS:
            ds=h[key]; digest=hashlib.sha256()
            for i in range(0,ds.shape[0],64): digest.update(np.asarray(ds[i:i+64]).tobytes(order='C'))
            result[key]=[list(ds.shape),str(ds.dtype),digest.hexdigest()]
    return result

def analyze(protocol_path:Path,campaign_root:Path,validate=False):
    p=_read(protocol_path); errors=[]
    jobs_path=protocol_path.parent/p['jobs']['path']; jobs=_read(jobs_path)
    delta=(SOURCE['inner2']-SOURCE['inner1'])/4
    frozen=dict(count=24,steps_per_job=5,segment_horizon=5000,snapshot_interval=1,
                skip_ratio=.6,prior_cumulative_horizon=50000,total_segments=120,total_tau=600000)
    if p.get('schema')!='tdgl-full-curl-local-response-plan-v1': errors.append('schema mismatch')
    if p.get('source_currents')!=SOURCE or p.get('seeds')!=list(SEEDS) or p.get('arms')!=dict(minus=-1,sham=0,plus=1) or not close(p.get('delta'),delta): errors.append('frozen source/seed/arm/delta mismatch')
    if any(p['jobs'].get(k)!=v for k,v in frozen.items()): errors.append('frozen job protocol mismatch')
    if p['jobs'].get('jobs_sha256')!=_sha(jobs_path): errors.append('jobs hash mismatch')
    if p.get('source_hashes')!=dict(config=CONFIG_HASH,model=MODEL_HASH,rhs_source=RHS_HASH): errors.append('source hash mismatch')
    guard=dict(maxiters_override=10000000,fail_fast=True,source_sha256=prep.GUARD_HASH)
    if p.get('solver_guard')!=guard: errors.append('solver guard mismatch')
    if p.get('backend')!='cpu_compatible' or p.get('tolerance')!=1e-4: errors.append('solver protocol mismatch')
    if p.get('confidence')!=dict(bonferroni_contrasts=4,stationarity_ci_abs_max=.0025,negative_delta_threshold=-.01): errors.append('confidence rules mismatch')
    if p.get('final_window')!=[18000,20000] or p.get('drift_windows')!=[[13000,15000],[18000,20000]]: errors.append('analysis windows mismatch')
    expected={(label,arm,seed) for label in SOURCE for arm in ARMS for seed in SEEDS}
    actual={(j.get('source_label'),j.get('arm'),j.get('seed')) for j in jobs}; ids={j.get('id') for j in jobs}
    if len(jobs)!=24 or len(ids)!=24 or actual!=expected: errors.append('exact source/arm/seed matrix mismatch')
    sources={(s.get('source_label'),s.get('seed')):s for s in p.get('sources',[])}
    if len(p.get('sources',[]))!=8 or set(sources)!={(l,s) for l in SOURCE for s in SEEDS}: errors.append('source receipt matrix mismatch')
    for key,source in sources.items():
        try:
            label,seed=key
            for path_key,hash_key in [('hdf5_path','output_hash'),('sidecar_path','sidecar_sha256'),('rng_path','rng_sha256'),('chain_path','chain_sha256')]:
                if _sha(source[path_key])!=source[hash_key]: errors.append(f'source hash mismatch {key}/{path_key}')
            e=_read(source['chain_path']); m=e['manifest']
            if (label not in SOURCE or seed not in SEEDS or not close(source.get('Je'),SOURCE.get(label,math.nan))
                or e.get('seed')!=seed or m.get('seed')!=seed or e.get('segment')!=9
                or e.get('cumulative_start')!=45000 or e.get('cumulative_end')!=50000
                or not close(e.get('Je'),source['Je']) or not close(m.get('Je'),source['Je'])
                or m.get('output_path')!=source['hdf5_path'] or m.get('output_hash')!=source['output_hash']
                or e.get('rng_after_path')!=source['rng_path'] or e.get('rng_after_sha256')!=source['rng_sha256']
                or _read(source['sidecar_path'])!=m
                or any(m.get(k)!=v for k,v in [('config_hash',CONFIG_HASH),('model_hash',MODEL_HASH),('rhs_source_hash',RHS_HASH)])
                or m.get('complete') is not True or m.get('status')!='success'):
                errors.append(f'source receipt identity mismatch {key}')
        except Exception as exc: errors.append(f'source validation failure {key}: {exc}')
    grouped=defaultdict(list)
    for path in campaign_root.glob('**/step*/chain.json'):
        try:
            e=_read(path);e['_chain_path']=str(path);e['_segment_dir']=str(path.parent);grouped[str(e.get('id'))].append(e)
        except Exception as exc: errors.append(f'malformed receipt {path}: {exc}')
    if set(grouped)!=ids: errors.append('campaign job IDs mismatch')
    rows=[];segment_rows=[];conditioning={}
    for job in jobs:
        jid,label,arm,seed=job.get('id'),job.get('source_label'),job.get('arm'),job.get('seed')
        if (label,arm,seed) not in expected: continue
        source=sources.get((label,seed));target=SOURCE[label]+dict(minus=-1,sham=0,plus=1)[arm]*delta
        sequence=[SOURCE[label]]+[target]*4
        if (jid!=f'{label}_{arm}_seed{seed}' or job.get('currents')!=sequence
            or not close(job.get('source_Je'),SOURCE[label]) or not close(job.get('target_Je'),target)
            or job.get('prior_cumulative_horizon')!=50000 or job.get('Nx')!=60 or job.get('Ny')!=30 or job.get('noise')!=.001
            or source is None or job.get('initial_state_path')!=source['hdf5_path'] or job.get('initial_rng_path')!=source['rng_path']): errors.append(f'job declaration mismatch {jid}')
        try:
            if _sha(job['config_path'])!=CONFIG_HASH: errors.append(f'job config mismatch {jid}')
        except Exception as exc: errors.append(f'job config failure {jid}: {exc}')
        es=sorted(grouped.get(jid,[]),key=lambda e:int(e.get('step',0)))
        if len(es)!=5 or [e.get('step') for e in es]!=[1,2,3,4,5]: errors.append(f'missing steps {jid}');continue
        means=[]
        for i,e in enumerate(es,1):
            m=e.get('manifest',{})
            numeric=dict(horizon=5000,final_time=5000,snapshot_interval=1,skip_ratio=.6,tolerance=1e-4,
                         noise=.001,Nx=60,Ny=30,hx=.5,hy=.5,Je=sequence[i-1],seed=seed)
            if (any(not close(m.get(k),v) for k,v in numeric.items())
                or e.get('seed')!=seed or not close(e.get('Je'),sequence[i-1])
                or e.get('cumulative_start')!=50000+(i-1)*5000 or e.get('cumulative_end')!=50000+i*5000
                or m.get('complete') is not True or m.get('status')!='success'
                or m.get('model_hash')!=MODEL_HASH or m.get('rhs_source_hash')!=RHS_HASH or m.get('config_hash')!=CONFIG_HASH
                or m.get('backend')!='cpu_compatible' or m.get('solver')!='Tsit5' or m.get('precision') not in ('Float64','ComplexF64/Float64')
                or m.get('save_fields') is not True or m.get('output_schema')!='streamed_hdf5_fields'
                or m.get('stream_maxiters_override')!=guard['maxiters_override'] or m.get('stream_guard_source_hash')!=guard['source_sha256']): errors.append(f'metadata mismatch {jid}/{i}')
            try:
                prev_h5=job['initial_state_path'] if i==1 else es[i-2]['manifest']['output_path']
                prev_rng=job['initial_rng_path'] if i==1 else es[i-2]['rng_after_path']
                if e.get('preceding_hdf5')!=prev_h5 or m.get('initial_state_source')!=prev_h5 or _sha(prev_h5)!=m.get('initial_source_hash'): errors.append(f'HDF5 continuity mismatch {jid}/{i}')
                if e.get('preceding_rng')!=prev_rng or _sha(prev_rng)!=e.get('rng_before_sha256'): errors.append(f'RNG continuity mismatch {jid}/{i}')
                if _sha(e['rng_after_path'])!=e.get('rng_after_sha256'): errors.append(f'RNG hash mismatch {jid}/{i}')
                if _sha(m['output_path'])!=m.get('output_hash'): errors.append(f'output hash mismatch {jid}/{i}')
                if _sha(m['initial_vector_path'])!=m.get('initial_vector_hash'): errors.append(f'initial vector hash mismatch {jid}/{i}')
                if validate:
                    ae,_=validate_artifact(Path(m['output_path']).with_suffix('.json'),m)
                    errors.extend(f'{jid}/{i}: {x}' for x in ae)
                    if not ae: errors.extend(f'{jid}/{i}: {x}' for x in _finite_fields_errors([e]))
                data=read_segment(e)
                if len(data['times'])!=2001 or not np.allclose(data['times'],np.arange(3000,5001),rtol=0,atol=1e-10): errors.append(f'schedule mismatch {jid}/{i}')
                vals={k:float(np.mean(data[k])) for k in ('V','T_avg','psi_avg')};vals['V_integrated']=float(data['V_integrated'])
                if any(not math.isfinite(v) for v in vals.values()): errors.append(f'nonfinite observables {jid}/{i}')
                means.append(vals)
                segment_rows.append(dict(job_id=jid,source_label=label,arm=arm,seed=seed,step=i,Je=sequence[i-1],
                    response_window=None if i==1 else [(i-2)*5000+3000,(i-1)*5000],**vals))
                if i==1:
                    conditioning[(label,arm,seed)]=dict(datasets=dataset_digests(m['output_path']),initial=m['initial_vector_hash'],
                        rng_before=e['rng_before_sha256'],rng_after=e['rng_after_sha256'],accepted=m.get('accepted_steps'),rejected=m.get('rejected_steps'),rhs_calls=m.get('rhs_calls'))
            except Exception as exc: errors.append(f'artifact/read failure {jid}/{i}: {exc}')
        if len(means)==5: rows.append(dict(job_id=jid,source_label=label,arm=arm,seed=seed,target_Je=target,drift=means[-1]['V']-means[-2]['V'],**means[-1]))
    conditioning_ok=len(conditioning)==24
    for label in SOURCE:
        for seed in SEEDS:
            entries=[conditioning.get((label,arm,seed)) for arm in ARMS]
            if any(x is None for x in entries) or not all(x==entries[0] for x in entries): errors.append(f'conditioning identity mismatch {label}/{seed}');conditioning_ok=False
    by={(r['source_label'],r['arm'],r['seed']):r for r in rows};contrasts={};stationary={}
    for label in SOURCE:
        contrasts[label]={}
        for key,lo,hi in [('sham_minus','minus','sham'),('plus_minus_sham','sham','plus'),('centered_plus_minus','minus','plus')]:
            values=[by[(label,hi,s)]['V']-by[(label,lo,s)]['V'] for s in SEEDS if (label,lo,s) in by and (label,hi,s) in by]
            contrasts[label][key]=paired(values)
        stationary[label]={arm:drift_stats([r['drift'] for r in rows if r['source_label']==label and r['arm']==arm]) for arm in ARMS}
    complete=not errors and len(rows)==24 and len(segment_rows)==120;verified=complete and validate
    for label in SOURCE:
        for key,value in contrasts[label].items(): value['negative_candidate']=bool(verified and key!='centered_plus_minus' and value['n']==4 and value['upper95_bonferroni4']<-.01)
        for value in stationary[label].values(): value['pass_drift']=bool(verified and value['pass_drift'])
    code=[Path(__file__),Path(prep.__file__),Path(__file__).with_name('analyze_full_curl_ndr.py'),Path(__file__).with_name('analyze_full_iv.py'),Path(__file__).with_name('compare_results.py')]
    return _finite(dict(schema='tdgl-full-curl-local-response-analysis-v1',complete=complete,pending=not complete,errors=errors,
        artifact_validation=validate,finite_saved_fields_validation=verified,protocol=p,rows=rows,segment_rows=segment_rows,
        contrasts=contrasts,stationarity=stationary,conditioning_identity_checked=bool(verified and conditioning_ok),
        conditioning_digests={f'{l}/{a}/{s}':v for (l,a,s),v in conditioning.items()},stationary_branch_confirmed=False,
        provenance=dict(protocol_sha256=_sha(protocol_path),jobs_sha256=_sha(jobs_path),analysis_source_sha256={x.name:_sha(x) for x in code},
            chain_receipt_sha256={e['_chain_path']:_sha(e['_chain_path']) for es in grouped.values() for e in es},python=sys.version,numpy=np.__version__,scipy=scipy.__version__,h5py=h5py.__version__),
        claim='Exploratory local step responses from reused prepared seeds; no continuous stationary-branch claim.'))

def write_outputs(result,out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'state_response_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    for name,key in [('state_response_rows.csv','rows'),('state_response_segment_rows.csv','segment_rows')]:
        if not result[key]: continue
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(result[key][0]));writer.writeheader();writer.writerows(result[key])
    if not result['complete']: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(18,9),constrained_layout=True)
    colors=dict(minus='#0072B2',sham='#333333',plus='#D55E00')
    for row,label in enumerate(SOURCE):
        for arm in ARMS:
            means=[];half=[]
            for step in range(2,6):
                vals=[r['V'] for r in result['segment_rows'] if r['source_label']==label and r['arm']==arm and r['step']==step]
                means.append(np.mean(vals));half.append(stats.t.ppf(.975,3)*np.std(vals,ddof=1)/2)
            axes[row,0].errorbar([4000,9000,14000,19000],means,yerr=half,fmt='o-',color=colors[arm],label=arm,capsize=3)
            for seed in SEEDS:
                vals=[r['V'] for r in result['segment_rows'] if r['source_label']==label and r['arm']==arm and r['seed']==seed and r['step']>1]
                axes[row,0].plot([4000,9000,14000,19000],vals,color=colors[arm],alpha=.22,lw=.7)
        for col,key in [(1,'V'),(2,'T_avg'),(3,'psi_avg')]:
            values=[[r[key] for r in result['rows'] if r['source_label']==label and r['arm']==a] for a in ARMS]
            for seed in SEEDS:
                axes[row,col].plot([-1,0,1],[next(r[key] for r in result['rows'] if r['source_label']==label and r['arm']==a and r['seed']==seed) for a in ARMS],color='#0072B2',alpha=.25,lw=.7)
            axes[row,col].errorbar([-1,0,1],[np.mean(v) for v in values],yerr=[stats.t.ppf(.975,3)*np.std(v,ddof=1)/2 for v in values],fmt='o-',capsize=4)
            axes[row,col].set_xticks([-1,0,1],['−ΔJ','Sham','+ΔJ']);axes[row,col].set(xlabel='Current step from common prepared state',ylabel={'V':'V','T_avg':'Mean T','psi_avg':'Mean |ψ|²'}[key])
        axes[row,0].set(xlabel='Response-window midpoint after current step (τ)',ylabel='V');axes[row,0].legend()
        for axis in axes[row]: axis.grid(alpha=.2);axis.set_title(f'{label} · source Je={SOURCE[label]:.6f}')
    fig.suptitle('Full-curl local current-step responses · T₀=0.18\nFour reused seeds; nominal 95% seed CI; late response 18–20kτ\nLines guide comparisons between forked responses; stationary branch unconfirmed')
    fig.savefig(out/'state_response.png',dpi=180);fig.savefig(out/'state_response.pdf');plt.close(fig)

def main(argv=None):
    parser=argparse.ArgumentParser()
    for flag in ['protocol','campaign-root','out']: parser.add_argument('--'+flag,required=True,type=Path)
    parser.add_argument('--validate-artifacts',action='store_true');args=parser.parse_args(argv)
    result=analyze(args.protocol,args.campaign_root,args.validate_artifacts);write_outputs(result,args.out)
    print(json.dumps(dict(complete=result['complete'],errors=result['errors'][:12],out=str(args.out))))
    return 0 if result['complete'] else 1

if __name__=='__main__': raise SystemExit(main())
