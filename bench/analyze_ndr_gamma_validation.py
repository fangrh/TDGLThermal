#!/usr/bin/env python3
"""Eight-seed matched gamma comparisons under prospectively amended v2."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import t
import yaml
sys.path.insert(0,str(Path(__file__).parent))
import prepare_ndr_gamma_tune as prep
import prepare_ndr_gamma_validation as plan
from analyze_ndr_enhancement import _protocol_ok
from analyze_full_curl_ndr import read_segment,validate_chain_artifacts
from analyze_full_iv import _finite_fields_errors
from analyze_ndr_gamma_tune import sha,read,finite

def get_confidence(values):
    n=len(values);mean=float(np.mean(values)) if n else math.nan
    sem=float(np.std(values,ddof=1)/math.sqrt(n)) if n>1 else math.nan
    crit=float(t.ppf(.975,n-1)) if n>1 else math.nan
    return dict(n=n,values=values,mean=mean,sem=sem,crit=crit,lower=mean-crit*sem,upper=mean+crit*sem)

def analyze(protocol_path:Path,campaign_root:Path,validate=False):
    p=read(protocol_path);errors=[];jobs_path=protocol_path.parent/p['jobs']['path'];jobs=read(jobs_path)
    currents=prep.CURRENTS[1:3];seeds=plan.SEEDS;variants=[v for v,_ in prep.VARIANTS]
    if p.get('schema')!=plan.SCHEMA:errors.append('validation schema mismatch')
    if p.get('currents')!=currents or p.get('seeds')!=seeds or p.get('variants')!=[{'id':v,'gamma':g,'config':f'configs/{v}.yaml'} for v,g in prep.VARIANTS]:errors.append('frozen gamma/current/seed matrix mismatch')
    if p.get('primary')!={'Je_low':currents[0],'Je_high':currents[1]}:errors.append('fixed primary mismatch')
    if p.get('source_hashes')!={'model':prep.MODEL_HASH,'rhs':prep.RHS_HASH}:errors.append('physics source mismatch')
    if p.get('solver_guard')!={'maxiters_override':10000000,'source_sha256':prep.GUARD_HASH,'fail_fast':True}:errors.append('guard protocol mismatch')
    if p.get('success')!=plan.SUCCESS or p.get('winner_rule')!=plan.WINNER_RULE:errors.append('success/winner rule mismatch')
    if (p.get('alpha_family')!=.05 or p.get('alpha_per_candidate')!=.025 or p.get('candidate_comparisons')!=2
        or p.get('early_stop') is not False or p.get('total_budget_equivalent_segments')!=880):errors.append('alpha/allocation policy mismatch')
    if (p.get('backend')!='cpu_compatible' or p.get('tolerance')!=1e-4 or p.get('snapshot_interval')!=1 or p.get('skip_ratio')!=.6
        or p.get('final_window')!=[48000,50000] or p.get('drift_windows')!=[[43000,45000],[48000,50000]]
        or any(p['jobs'].get(k)!=v for k,v in {'count':48,'segments':10,'horizon':5000,'total_segments':480}.items())):errors.append('frozen timing/solver mismatch')
    if p['jobs'].get('jobs_sha256')!=sha(jobs_path):errors.append('jobs hash mismatch')
    try:
        sp=Path(p['screen_protocol_path']);ap=Path(p['amendment_path'])
        if sha(sp)!=p.get('screen_protocol_sha256') or sha(ap)!=p.get('amendment_sha256'):errors.append('amendment/screen hash mismatch')
        original,amendment=plan._require_screen(sp,ap);original_configs=plan._validate_configs(sp,original)
        if p.get('config_sha256')!=original.get('config_sha256'):errors.append('screen/validation config identities differ')
    except Exception as exc:errors.append(f'amendment lineage failure: {exc}');original_configs={}
    cfg_hashes={}
    for variant in variants:
        try:
            cp=protocol_path.parent/f'configs/{variant}.yaml';cfg_hashes[variant]=sha(cp)
            if cfg_hashes[variant]!=p.get('config_sha256',{}).get(variant) or cp.read_bytes()!=original_configs.get(variant):errors.append(f'config identity mismatch {variant}')
        except Exception as exc:errors.append(f'config failure {variant}: {exc}')
    expected={(v,je,s) for v in variants for je in currents for s in seeds};actual={(j.get('variant'),j.get('Je'),j.get('seed')) for j in jobs};ids={j.get('id') for j in jobs}
    if len(jobs)!=48 or len(ids)!=48 or actual!=expected:errors.append('incomplete/duplicated job matrix')
    grouped=defaultdict(list)
    for cp in campaign_root.glob('**/segment*/chain.json'):
        try:
            e=read(cp);e['_chain_path']=str(cp);e['_segment_dir']=str(cp.parent);grouped[str(e.get('id'))].append(e)
        except Exception as exc:errors.append(f'malformed receipt {cp}: {exc}')
    if set(grouped)!=ids:errors.append('receipt job coverage mismatch')
    rows=[];segment_rows=[];initial_groups=defaultdict(list);rng_groups=defaultdict(set)
    for job in jobs:
        jid,v,je,seed=job.get('id'),job.get('variant'),job.get('Je'),job.get('seed')
        if (v,je,seed) not in expected:continue
        cp=(protocol_path.parent/f'configs/{v}.yaml').resolve();idx=currents.index(je)
        if (jid!=f'{v}_i{idx:03d}_seed{seed}' or job.get('config_path')!=str(cp) or job.get('segments')!=10
            or job.get('Nx')!=60 or job.get('Ny')!=30 or job.get('noise')!=.001 or job.get('prior_cumulative_horizon')!=0
            or job.get('initial_state_path') is not None or job.get('initial_rng_path') is not None):errors.append(f'job settings/config mismatch {jid}')
        es=sorted(grouped.get(jid,[]),key=lambda e:int(e.get('segment',0)))
        if len(es)!=10 or [e.get('segment') for e in es]!=list(range(1,11)):errors.append(f'missing segments {jid}');continue
        values=[]
        for i,e in enumerate(es,1):
            m=e.get('manifest',{});errors.extend(f'{jid}/{i}: {x}' for x in _protocol_ok(e,job,p,cfg_hashes))
            if not math.isclose(float(m.get('Je',math.nan)),je,rel_tol=0,abs_tol=1e-12):errors.append(f'Je mismatch {jid}/{i}')
            if e.get('cumulative_start')!=(i-1)*5000 or e.get('cumulative_end')!=i*5000:errors.append(f'cumulative mismatch {jid}/{i}')
            if m.get('stream_maxiters_override')!=10000000 or m.get('stream_guard_source_hash')!=prep.GUARD_HASH:errors.append(f'guard metadata mismatch {jid}/{i}')
            if m.get('preparation')!='configured_down':errors.append(f'preparation mismatch {jid}/{i}')
            try:
                if sha(m['output_path'])!=m.get('output_hash'):errors.append(f'output hash mismatch {jid}/{i}')
                if sha(m['initial_vector_path'])!=m.get('initial_vector_hash'):errors.append(f'initial vector hash mismatch {jid}/{i}')
                if sha(e['rng_after_path'])!=e.get('rng_after_sha256'):errors.append(f'RNG hash mismatch {jid}/{i}')
                before=e.get('rng_before_sha256')
                if not isinstance(before,str) or len(before)!=64 or any(c not in '0123456789abcdef' for c in before):errors.append(f'RNG before invalid {jid}/{i}')
                if i==1:
                    if (any(e.get(k) is not None for k in ('preceding_hdf5','preceding_sha256','preceding_rng_path'))
                        or m.get('initial_state_source') is not None or m.get('initial_source_hash') is not None):errors.append(f'fresh initial source mismatch {jid}')
                    initial_groups[(je,seed)].append((v,m['initial_vector_hash'],before));rng_groups[seed].add(before)
                else:
                    prev=es[i-2];pm=prev['manifest']
                    if (e.get('preceding_hdf5')!=pm['output_path'] or e.get('preceding_sha256')!=pm['output_hash']
                        or m.get('initial_state_source')!=pm['output_path'] or m.get('initial_source_hash')!=pm['output_hash']):errors.append(f'initial HDF continuity mismatch {jid}/{i}')
                    if e.get('preceding_rng_path')!=prev['rng_after_path'] or before!=prev['rng_after_sha256'] or sha(e['preceding_rng_path'])!=before:errors.append(f'RNG continuity mismatch {jid}/{i}')
                data=read_segment(e)
                if len(data['times'])!=2001 or not np.allclose(data['times'],np.arange(3000,5001),rtol=0,atol=1e-10):errors.append(f'retained schedule mismatch {jid}/{i}')
                item={k:float(np.mean(data[k])) for k in ('V','T_avg','psi_avg')};item['V_integrated']=float(data['V_integrated'])
                if any(not math.isfinite(x) for x in item.values()):errors.append(f'nonfinite observables {jid}/{i}')
                values.append(item);segment_rows.append(dict(job_id=jid,variant=v,Je=je,seed=seed,segment=i,**item))
            except Exception as exc:errors.append(f'artifact/read failure {jid}/{i}: {exc}')
        if validate:
            ae=validate_chain_artifacts(es);errors.extend(f'{jid}: {x}' for x in ae)
            if not ae:errors.extend(f'{jid}: {x}' for x in _finite_fields_errors(es))
        if len(values)==10:
            totals={k:sum(int(e['manifest'].get(k) or 0) for e in es) for k in ('accepted_steps','rejected_steps','rhs_calls')}
            rows.append(dict(job_id=jid,variant=v,Je=je,seed=seed,drift=values[-1]['V']-values[-2]['V'],**values[-1],solver_totals=totals))
    initial_ok=len(initial_groups)==16
    for key,items in initial_groups.items():
        if len(items)!=3 or len({v for v,_,_ in items})!=3 or len({(iv,rng) for _,iv,rng in items})!=1:errors.append(f'fresh initial identity mismatch {key}');initial_ok=False
    if any(len(x)!=1 for x in rng_groups.values()):errors.append('fresh RNG identity mismatch across current/config')
    stationary={v:{} for v in variants};by={(r['variant'],r['Je'],r['seed']):r for r in rows};pairs={}
    for v in variants:
        for je in currents:
            d=get_confidence([r['drift'] for r in rows if r['variant']==v and r['Je']==je]);d['pass_drift']=bool(d['n']==8 and d['lower']>=-.0025 and d['upper']<=.0025);stationary[v][str(je)]=d
        pairs[v]=[]
        for seed in seeds:
            lo=by.get((v,currents[0],seed));hi=by.get((v,currents[1],seed))
            if lo is not None and hi is not None:pairs[v].append(dict(seed=seed,V_low=lo['V'],V_high=hi['V'],fraction=(lo['V']-hi['V'])/lo['V'] if lo['V'] else math.nan,deltaV=hi['V']-lo['V']))
    baseline={x['seed']:x['fraction'] for x in pairs['baseline']}
    complete=not errors and len(rows)==48 and len(segment_rows)==480;verified=complete and validate;comparisons={}
    for v in variants:
        fractions=get_confidence([x['fraction'] for x in pairs[v]]);dv=get_confidence([x['deltaV'] for x in pairs[v]])
        improvement=get_confidence([x['fraction']-baseline.get(x['seed'],math.nan) for x in pairs[v]])
        gates=dict(all_eight_positive=len(pairs[v])==8 and all(x['fraction']>0 for x in pairs[v]),mean_fraction_at_least_20=fractions['mean']>=.20,
                   fraction_lower_above_15=fractions['lower']>.15,paired_improvement_lower_positive=improvement['lower']>0,deltaV_upper_below_minus_point01=dv['upper']<-.01,
                   endpoint_drift=all(stationary[x][str(je)]['pass_drift'] for x in ('baseline',v) for je in currents))
        comparisons[v]=dict(seeds=pairs[v],fraction=fractions,deltaV=dv,paired_improvement=improvement,gates=gates,accepted=bool(verified and v!='baseline' and all(gates.values())))
    candidates=[v for v in ('gamma38','gamma42') if comparisons[v]['accepted']]
    winner=max(candidates,key=lambda v:(comparisons[v]['paired_improvement']['lower'],comparisons[v]['fraction']['mean'],v=='gamma38')) if candidates else None
    if not verified:
        for values in stationary.values():
            for d in values.values():d['pass_drift']=False
        for comparison in comparisons.values():
            comparison['gates']={key:False for key in comparison['gates']}
    helpers=[Path(__file__),Path(plan.__file__),Path(prep.__file__),Path(__file__).with_name('analyze_ndr_gamma_tune.py'),Path(__file__).with_name('analyze_ndr_enhancement.py'),Path(__file__).with_name('analyze_full_curl_ndr.py'),Path(__file__).with_name('analyze_full_iv.py'),Path(__file__).with_name('compare_results.py')]
    return finite(dict(schema='tdgl-full-curl-gamma-fresh-validation-analysis-v2',complete=complete,pending=not complete,errors=errors,
        artifact_validation=validate,finite_saved_fields_validation=verified,fresh_initial_identity_checked=bool(verified and initial_ok),protocol=p,
        rows=rows,segment_rows=segment_rows,comparisons=comparisons,stationarity=stationary,selected_variant=winner,stationary_branch_confirmed=False,
        provenance=dict(protocol_sha256=sha(protocol_path),jobs_sha256=sha(jobs_path),analysis_source_sha256={x.name:sha(x) for x in helpers},chain_receipt_sha256={e['_chain_path']:sha(e['_chain_path']) for es in grouped.values() for e in es},python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pyyaml=yaml.__version__),
        scope='Prospectively amended v2 endpoint validation; mean seedwise fractions, paired comparisons with Bonferroni2 one-sided alpha .025. Continuous stationary branch unconfirmed.'))

def write_outputs(r,out):
    out.mkdir(parents=True,exist_ok=True);(out/'gamma_validation_analysis.json').write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8')
    for key,name in [('rows','gamma_validation_rows.csv'),('segment_rows','gamma_validation_segments.csv')]:
        if not r[key]:continue
        with (out/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(r[key][0]));w.writeheader();w.writerows(r[key])
    if not (r['complete'] and r['artifact_validation'] and r['finite_saved_fields_validation'] and r['fresh_initial_identity_checked']):return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained');colors={'baseline':'#333333','gamma38':'#0072B2','gamma42':'#D55E00'};currents=r['protocol']['currents'];cats=[];di=[];dh=[]
    for v,gamma in prep.VARIANTS:
        values=np.array([[next(x['V'] for x in r['rows'] if x['variant']==v and x['Je']==je and x['seed']==seed) for je in currents] for seed in plan.SEEDS])
        for row in values:axes[0,0].plot(currents,row,'o-',color=colors[v],alpha=.18,lw=.7,ms=3)
        axes[0,0].errorbar(currents,values.mean(axis=0),yerr=t.ppf(.975,7)*values.std(axis=0,ddof=1)/np.sqrt(8),fmt='o-',color=colors[v],capsize=4,lw=2,label=f'γ={gamma:g}')
        c=r['comparisons'][v];d=c['fraction'];axes[0,1].plot([gamma]*8,[100*x['fraction'] for x in c['seeds']],'o',color=colors[v],ms=4,alpha=.6)
        axes[0,1].errorbar(gamma,100*d['mean'],yerr=100*d['crit']*d['sem'],fmt='s',color=colors[v],capsize=5,ms=7)
        if v!='baseline':
            d=c['paired_improvement'];axes[1,0].plot([gamma]*8,[100*x for x in d['values']],'o',color=colors[v],alpha=.6,ms=4)
            axes[1,0].errorbar(gamma,100*d['mean'],yerr=100*d['crit']*d['sem'],fmt='s',color=colors[v],capsize=5,ms=7)
        for je,side in zip(currents,['low','high']):
            d=r['stationarity'][v][str(je)];cats.append(f'γ{gamma:g}\n{side} J');di.append(d['mean']);dh.append(d['crit']*d['sem'])
    axes[0,0].set(xlabel='Je (model units)',ylabel='V (model units)',title='Prepared endpoints · mean ± nominal 95% CI');axes[0,0].legend()
    axes[0,1].axhline(20,color='#777777',ls='--',lw=1);axes[0,1].set(xlabel='Gamma',ylabel='Paired voltage decrease (%)',xticks=[38,40,42],title='Eight fresh seeds per gamma')
    axes[1,0].axhline(0,color='#777777',ls='--',lw=1);axes[1,0].set(xlabel='Gamma',ylabel='Improvement over γ40 (percentage points)',xticks=[38,42],title='Paired comparisons · one-sided α=0.025 bounds')
    axes[1,1].axhspan(-.0025,.0025,color='#009E73',alpha=.12);axes[1,1].errorbar(range(6),di,yerr=dh,fmt='o',color='#333333',capsize=4);axes[1,1].set(xticks=range(6),xticklabels=cats,ylabel='Late-window voltage drift',title='Endpoint drift · nominal 95% CI')
    for axis in axes.flat:axis.grid(alpha=.2)
    fig.suptitle('Full-curl gamma microtuning · fresh eight-seed validation v2\n50kτ · final 48–50kτ · independent prepared points; stationary branch unconfirmed')
    fig.savefig(out/'gamma_validation.png',dpi=180);fig.savefig(out/'gamma_validation.pdf');plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser()
    for key in ('protocol','campaign-root','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--validate-artifacts',action='store_true');a=p.parse_args(argv);r=analyze(a.protocol,a.campaign_root,a.validate_artifacts);write_outputs(r,a.out)
    print(json.dumps({'complete':r['complete'],'errors':r['errors'][:12],'selected_variant':r['selected_variant']}));return 0 if r['complete'] else 1

if __name__=='__main__':raise SystemExit(main())
