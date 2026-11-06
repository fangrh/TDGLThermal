#!/usr/bin/env python3
"""Matched, fixed-bracket analysis of the gamma 38/40/42 full-curl screen."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import t
import yaml
sys.path.insert(0,str(Path(__file__).parent))
import prepare_ndr_gamma_tune as prep
from analyze_ndr_enhancement import _protocol_ok, _pair_rows
from analyze_full_curl_ndr import read_segment, validate_chain_artifacts
from analyze_full_iv import _finite_fields_errors

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def finite(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite(v) for v in value]
    return value

def interval(values):
    n=len(values);mean=float(np.mean(values)) if n else math.nan
    half=float(t.ppf(.975,n-1)*np.std(values,ddof=1)/math.sqrt(n)) if n>1 else math.nan
    return dict(n=n,values=values,mean=mean,ci_low=mean-half,ci_high=mean+half,
                pass_drift=bool(n==2 and abs(mean)+half<=.0025))

def analyze(protocol_path:Path,campaign_root:Path,validate=False):
    p=read(protocol_path);jobs_path=protocol_path.parent/p['jobs']['path'];jobs=read(jobs_path);errors=[]
    if p.get('schema')!='tdgl-full-curl-gamma-fine-tune-screen-v1':errors.append('protocol schema mismatch')
    expected_variants=[{'id':v,'gamma':g,'config':f'configs/{v}.yaml'} for v,g in prep.VARIANTS]
    if p.get('variants')!=expected_variants or p.get('currents')!=prep.CURRENTS or p.get('seeds')!=prep.SEEDS:errors.append('frozen variant/current/seed matrix mismatch')
    if p.get('source_hashes')!={'model':prep.MODEL_HASH,'rhs':prep.RHS_HASH}:errors.append('physics source mismatch')
    guard={'maxiters_override':10000000,'source_sha256':prep.GUARD_HASH,'fail_fast':True}
    if p.get('solver_guard')!=guard:errors.append('guard protocol mismatch')
    if (any(p.get('jobs',{}).get(k)!=v for k,v in {'count':24,'segments':10,'horizon':5000,'total_segments':240}.items())
        or p.get('snapshot_interval')!=1 or p.get('skip_ratio')!=.6 or p.get('backend')!='cpu_compatible' or p.get('tolerance')!=1e-4
        or p.get('final_window')!=[48000,50000] or p.get('drift_windows')!=[[43000,45000],[48000,50000]]):errors.append('frozen timing/solver mismatch')
    primary={'Je_low':prep.CURRENTS[1],'Je_high':prep.CURRENTS[2]}
    if p.get('primary')!=primary:errors.append('fixed primary bracket mismatch')
    eligibility={'positive_drop_each_seed':True,'positive_improvement_each_seed':True,'absolute_drop_min':.01,'low_voltage_min':.05,'drift_ci_bound':.0025,'rank':'minimum paired fractional improvement over contemporaneous baseline'}
    if p.get('eligibility')!=eligibility:errors.append('eligibility rules mismatch')
    if p['jobs'].get('jobs_sha256')!=sha(jobs_path):errors.append('jobs hash mismatch')
    expected={(v,je,s) for v,_ in prep.VARIANTS for je in prep.CURRENTS for s in prep.SEEDS}
    actual={(j.get('variant'),j.get('Je'),j.get('seed')) for j in jobs};ids={j.get('id') for j in jobs}
    if len(jobs)!=24 or len(ids)!=24 or actual!=expected:errors.append('job matrix incomplete or duplicated')
    cfg_hashes={};baseline=None
    for variant,gamma in prep.VARIANTS:
        cp=protocol_path.parent/f'configs/{variant}.yaml'
        try:
            raw=cp.read_bytes();cfg_hashes[variant]=sha(cp)
            if cfg_hashes[variant]!=p.get('config_sha256',{}).get(variant):errors.append(f'config hash mismatch {variant}')
            parsed=yaml.safe_load(raw);prep._validate_baseline(parsed)
            if parsed['tdgl']['gamma']!=gamma:errors.append(f'gamma value mismatch {variant}')
            if variant=='baseline':
                baseline=raw
                if cfg_hashes[variant]!=prep.BASE_HASH:errors.append('baseline identity mismatch')
            elif baseline is None or raw!=prep._variant_bytes(baseline,gamma):errors.append(f'config changes outside gamma {variant}')
        except Exception as exc:errors.append(f'config validation failure {variant}: {exc}')
    grouped=defaultdict(list)
    for path in campaign_root.glob('**/segment*/chain.json'):
        try:
            e=read(path);e['_chain_path']=str(path);e['_segment_dir']=str(path.parent);grouped[str(e.get('id'))].append(e)
        except Exception as exc:errors.append(f'malformed receipt {path}: {exc}')
    if set(grouped)!=ids:errors.append('receipt job ID coverage mismatch')
    rows=[];segment_rows=[];initial_groups=defaultdict(list);seed_rng=defaultdict(set)
    for job in jobs:
        jid=job.get('id');variant=job.get('variant');je=job.get('Je');seed=job.get('seed')
        if (variant,je,seed) not in expected:continue
        idx=prep.CURRENTS.index(je);cp=protocol_path.parent/f'configs/{variant}.yaml'
        if (jid!=f'{variant}_i{idx:03d}_seed{seed}' or job.get('config_path')!=str(cp.resolve()) or job.get('segments')!=10
            or job.get('Nx')!=60 or job.get('Ny')!=30 or job.get('noise')!=.001 or job.get('prior_cumulative_horizon')!=0
            or job.get('initial_state_path') is not None or job.get('initial_rng_path') is not None):errors.append(f'job settings/config mismatch {jid}')
        es=sorted(grouped.get(jid,[]),key=lambda e:int(e.get('segment',0)))
        if len(es)!=10 or [e.get('segment') for e in es]!=list(range(1,11)):errors.append(f'missing segments {jid}');continue
        means=[]
        for i,e in enumerate(es,1):
            m=e.get('manifest',{})
            errors.extend(f'{jid}/{i}: {x}' for x in _protocol_ok(e,job,p,cfg_hashes))
            if not math.isclose(float(m.get('Je',math.nan)),float(je),rel_tol=0,abs_tol=1e-12):errors.append(f'Je metadata mismatch {jid}/{i}')
            if e.get('cumulative_start')!=(i-1)*5000 or e.get('cumulative_end')!=i*5000:errors.append(f'cumulative continuity mismatch {jid}/{i}')
            if m.get('stream_maxiters_override')!=10000000 or m.get('stream_guard_source_hash')!=prep.GUARD_HASH:errors.append(f'guard metadata mismatch {jid}/{i}')
            if m.get('preparation')!='configured_down':errors.append(f'preparation mismatch {jid}/{i}')
            try:
                if sha(m['output_path'])!=m.get('output_hash'):errors.append(f'output hash mismatch {jid}/{i}')
                if not Path(m['initial_vector_path']).is_file():errors.append(f'missing initial vector {jid}/{i}')
                elif sha(m['initial_vector_path'])!=m.get('initial_vector_hash'):errors.append(f'initial vector hash mismatch {jid}/{i}')
                if sha(e['rng_after_path'])!=e.get('rng_after_sha256'):errors.append(f'RNG hash mismatch {jid}/{i}')
                before=e.get('rng_before_sha256')
                if not isinstance(before,str) or len(before)!=64 or any(c not in '0123456789abcdef' for c in before):errors.append(f'RNG before hash invalid {jid}/{i}')
                if i==1:
                    if any(e.get(k) is not None for k in ('preceding_hdf5','preceding_sha256','preceding_rng_path')) or m.get('initial_state_source') is not None or m.get('initial_source_hash') is not None:errors.append(f'fresh initial source mismatch {jid}')
                    initial_groups[(je,seed)].append((variant,m.get('initial_vector_hash'),before));seed_rng[seed].add(before)
                else:
                    prev=es[i-2];pm=prev['manifest']
                    if (e.get('preceding_hdf5')!=pm['output_path'] or e.get('preceding_sha256')!=pm['output_hash']
                        or m.get('initial_state_source')!=pm['output_path'] or m.get('initial_source_hash')!=pm['output_hash']):errors.append(f'initial HDF continuity mismatch {jid}/{i}')
                    if (e.get('preceding_rng_path')!=prev['rng_after_path'] or before!=prev['rng_after_sha256']
                        or sha(e['preceding_rng_path'])!=before):errors.append(f'RNG continuity mismatch {jid}/{i}')
                data=read_segment(e)
                if len(data['times'])!=2001 or not np.allclose(data['times'],np.arange(3000,5001),rtol=0,atol=1e-10):errors.append(f'retained schedule mismatch {jid}/{i}')
                values={k:float(np.mean(data[k])) for k in ('V','T_avg','psi_avg')};values['V_integrated']=float(data['V_integrated'])
                if any(not math.isfinite(x) for x in values.values()):errors.append(f'nonfinite observables {jid}/{i}')
                means.append(values);segment_rows.append(dict(job_id=jid,variant=variant,Je=je,seed=seed,segment=i,**values))
            except Exception as exc:errors.append(f'artifact/read failure {jid}/{i}: {exc}')
        if validate:
            ae=validate_chain_artifacts(es);errors.extend(f'{jid}: {x}' for x in ae)
            if not ae:errors.extend(f'{jid}: {x}' for x in _finite_fields_errors(es))
        if len(means)==10:
            sums={k:sum(int(e['manifest'].get(k) or 0) for e in es) for k in ('accepted_steps','rejected_steps','rhs_calls')}
            rows.append(dict(job_id=jid,variant=variant,Je=je,seed=seed,drift=means[-1]['V']-means[-2]['V'],**means[-1],solver_totals=sums))
    initial_ok=len(initial_groups)==8
    for key,items in initial_groups.items():
        if len(items)!=3 or len({v for v,_,_ in items})!=3 or len({(iv,rng) for _,iv,rng in items})!=1:errors.append(f'fresh initial identity mismatch {key}');initial_ok=False
    if any(len(v)!=1 for v in seed_rng.values()):errors.append('fresh RNG identity mismatch across current/config')
    stationary={};by={(r['variant'],r['Je'],r['seed']):r for r in rows}
    for variant,_ in prep.VARIANTS:
        stationary[variant]={str(je):interval([r['drift'] for r in rows if r['variant']==variant and r['Je']==je]) for je in prep.CURRENTS}
    primary_result={**primary,'by_variant':{}};baseline_drops={}
    for variant,_ in prep.VARIANTS:
        vals=[]
        for seed in prep.SEEDS:
            lo=by.get((variant,primary['Je_low'],seed));hi=by.get((variant,primary['Je_high'],seed))
            if lo is None or hi is None:continue
            vl,vh=lo['V'],hi['V'];vals.append(dict(seed=seed,V_low=vl,V_high=vh,deltaV=vh-vl,fraction=(vl-vh)/vl if vl else math.nan))
        primary_result['by_variant'][variant]=dict(seeds=vals,mean_fraction=float(np.mean([x['fraction'] for x in vals])) if vals else math.nan)
        if variant=='baseline':baseline_drops={x['seed']:x['fraction'] for x in vals}
    complete=not errors and len(rows)==24 and len(segment_rows)==240;verified=complete and validate
    eligible=[]
    for variant,_ in prep.VARIANTS:
        item=primary_result['by_variant'][variant];vals=item['seeds']
        improvement=[x['fraction']-baseline_drops.get(x['seed'],math.nan) for x in vals]
        endpoints_pass=all(stationary[v][str(je)]['pass_drift'] for v in ('baseline',variant) for je in primary.values())
        qualifies=(len(vals)==2 and all(x['V_low']>=.05 and -x['deltaV']>.01 for x in vals)
            and len(improvement)==2 and all(x>0 for x in improvement) and endpoints_pass and variant!='baseline')
        item.update(paired_fractional_improvements=improvement,worst_seed_improvement=min(improvement) if improvement else math.nan,
                    endpoint_drift_pass=bool(verified and endpoints_pass),eligible=bool(verified and qualifies))
        if item['eligible']:eligible.append((item['worst_seed_improvement'],variant))
    winner=max(eligible,key=lambda x:(x[0],x[1]=='gamma38'))[1] if eligible else None
    if not verified:
        for points in stationary.values():
            for value in points.values():value['pass_drift']=False
    candidate=None if winner is None else dict(variant=winner,gamma=dict(prep.VARIANTS)[winner],**primary,
                                              worst_seed_improvement=primary_result['by_variant'][winner]['worst_seed_improvement'])
    helpers=[Path(__file__),Path(prep.__file__),Path(__file__).with_name('analyze_ndr_enhancement.py'),Path(__file__).with_name('analyze_full_curl_ndr.py'),Path(__file__).with_name('analyze_full_iv.py'),Path(__file__).with_name('compare_results.py')]
    return finite(dict(schema='tdgl-full-curl-gamma-fine-tune-analysis-v1',complete=complete,pending=not complete,errors=errors,
        artifact_validation=validate,finite_saved_fields_validation=verified,fresh_initial_identity_checked=bool(verified and initial_ok),
        protocol=p,rows=rows,segment_rows=segment_rows,stationarity=stationary,primary=primary_result,
        exploratory_adjacent_pairs=_pair_rows(rows,prep.CURRENTS,prep.SEEDS,[v for v,_ in prep.VARIANTS]),
        selected_variant=winner,selected_candidate=candidate,negative_branch=dict(stationary_claim=False),
        provenance=dict(protocol_sha256=sha(protocol_path),jobs_sha256=sha(jobs_path),analysis_source_sha256={x.name:sha(x) for x in helpers},
                        chain_receipt_sha256={e['_chain_path']:sha(e['_chain_path']) for es in grouped.values() for e in es},python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pyyaml=yaml.__version__),
        scope='Two-seed screening at the fixed original bracket. Other pairs exploratory. Continuous stationary branch unconfirmed.'))

def write_outputs(result,out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'gamma_screen_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    for key,name in [('rows','gamma_screen_rows.csv'),('segment_rows','gamma_screen_segments.csv')]:
        if not result[key]:continue
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            fields=list(result[key][0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(result[key])
    if not result['complete']:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained');colors={'baseline':'#333333','gamma38':'#0072B2','gamma42':'#D55E00'}
    for variant,gamma in prep.VARIANTS:
        for axis,key,title in [(axes[0,0],'V','Voltage'),(axes[1,0],'T_avg','Mean temperature'),(axes[1,1],'psi_avg','Mean |ψ|²')]:
            values=np.array([[next(r[key] for r in result['rows'] if r['variant']==variant and r['Je']==je and r['seed']==seed) for je in prep.CURRENTS] for seed in prep.SEEDS])
            for vals in values:axis.plot(prep.CURRENTS,vals,'o-',ms=3,lw=.8,color=colors[variant],alpha=.38)
            axis.plot(prep.CURRENTS,values.mean(axis=0),'o-',ms=5,lw=2,color=colors[variant],label=f'γ = {gamma:g}')
            axis.set(xlabel='Je (model units)',ylabel=title);axis.grid(alpha=.2)
        vals=result['primary']['by_variant'][variant]['seeds']
        for index,value in enumerate(vals):
            axes[0,1].plot(gamma,100*value['fraction'],marker=('o','s')[index],ms=6,color=colors[variant],linestyle='none')
    axes[0,0].axvspan(prep.CURRENTS[1],prep.CURRENTS[2],color='#777777',alpha=.08);axes[0,0].legend()
    axes[0,1].axhline(20,color='#777777',ls='--',lw=1,label='Conditional confirmation mean target')
    axes[0,1].set(xlabel='Gamma',ylabel='Primary paired voltage decrease (%)',xticks=[38,40,42]);axes[0,1].grid(alpha=.2);axes[0,1].legend(fontsize=8)
    fig.suptitle('Full-curl gamma microtuning · 50kτ · final 48–50kτ\nTwo fresh screening seeds per current; raw seed traces and means; stationary branch unconfirmed')
    fig.savefig(out/'gamma_screen.png',dpi=180);fig.savefig(out/'gamma_screen.pdf');plt.close(fig)

def main(argv=None):
    parser=argparse.ArgumentParser()
    for key in ('protocol','campaign-root','out'):parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--validate-artifacts',action='store_true');a=parser.parse_args(argv)
    result=analyze(a.protocol,a.campaign_root,a.validate_artifacts);write_outputs(result,a.out)
    print(json.dumps(dict(complete=result['complete'],errors=result['errors'][:12],candidate=result['selected_candidate'])))
    return 0 if result['complete'] else 1

if __name__=='__main__':raise SystemExit(main())
