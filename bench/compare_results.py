#!/usr/bin/env python3
"""Strict paired comparison of manifests and streamed HDF5 outputs."""
import argparse,csv,json,math,pathlib,sys
import h5py,numpy as np
REQ=('schema_version','role','backend','config_hash','model_hash','rhs_source_hash','julia_version','machine','initial_hash','seed','Je','noise','Nx','Ny','hx','hy','horizon','tolerance','snapshot_interval','skip_ratio','solver','precision','save_fields','status','complete','final_time','wall_seconds','output_path','output_hash','initial_vector_path','initial_vector_hash','output_schema')
FIELDS=('times','V','psi_avg','T_avg','psi_real','psi_imag','T','Ax','Ay')
def load(p):
    with open(p,encoding='utf-8') as f:return json.load(f)
def meta_errors(a,b):
    e=[]
    for x in (a,b):
        e += [f'missing {k}' for k in REQ if k not in x]
        if x.get('role') not in ('corrected_reference','compatible_candidate'):e.append(f'invalid role {x.get("role")}')
        if not x.get('complete',False):e.append(f'incomplete {x.get("backend")}')
        if x.get('status') not in ('success','Success','ReturnCode.Success'):e.append(f'failed status {x.get("backend")}')
        if not math.isfinite(float(x.get('final_time',float('nan')))):e.append(f'nonfinite final time {x.get("backend")}')
        if not math.isfinite(float(x.get('wall_seconds',float('nan')))) or float(x.get('wall_seconds',0))<=0:e.append(f'invalid wall time {x.get("backend")}')
    for k in ('config_hash','model_hash','rhs_source_hash','julia_version','machine','initial_hash','seed','Je','noise','Nx','Ny','hx','hy','horizon','tolerance','snapshot_interval','skip_ratio','solver','precision','save_fields','output_schema'):
        if a.get(k)!=b.get(k):e.append(f'mismatch {k}')
    for x in (a,b):
        if not math.isclose(float(x.get('final_time',np.nan)),float(x.get('horizon',np.nan)),abs_tol=1e-10,rel_tol=0):e.append(f'{x.get("backend")} horizon incomplete')
    if a.get('backend')!='reference' or a.get('role')!='corrected_reference':e.append('reference role/backend mismatch')
    if b.get('backend')!='cpu_compatible' or b.get('role')!='compatible_candidate':e.append('candidate role/backend mismatch')
    return e
def field_errors(rp,cp):
    e,v=[],{}
    with h5py.File(rp) as r,h5py.File(cp) as c:
        for k in FIELDS:
            if k not in r or k not in c:e.append(f'missing dataset {k}');continue
            a,b=np.asarray(r[k]),np.asarray(c[k])
            if a.shape!=b.shape:e.append(f'shape mismatch {k}');continue
            if not np.isfinite(a).all() or not np.isfinite(b).all():e.append(f'nonfinite dataset {k}');continue
            v[k]=float(np.max(np.abs(a-b))/(1+np.max(np.abs(a))))
    return e,v
def resolve_output(manifest_path, recorded):
    p=pathlib.Path(recorded); local=pathlib.Path(manifest_path).parent/p.name
    return p if p.is_file() else local
def validate_artifact(manifest_path, x):
    e=[]; out=resolve_output(manifest_path,x.get('output_path',''))
    if not out.is_file(): return [f'missing output artifact {out}'],None
    try:
        digest=__import__('hashlib').sha256(out.read_bytes()).hexdigest()
        if digest != x.get('output_hash'):e.append(f'output hash mismatch {x.get("backend")}')
        with h5py.File(out) as f:
            n=int(math.floor(float(x['horizon'])/float(x['snapshot_interval'])+1e-12))+1
            expected=np.arange(n,dtype=float)*float(x['snapshot_interval'])
            if not math.isclose(expected[-1],float(x['horizon']),abs_tol=1e-12,rel_tol=0):expected=np.append(expected,float(x['horizon']))
            skip=max(1,int(math.ceil(len(expected)*float(x['skip_ratio']))))
            expected=expected[skip-1:]
            ts=np.asarray(f['times'])
            if ts.ndim!=1 or len(ts)!=len(expected) or not np.all(np.diff(ts)>0) or not np.allclose(ts,expected,atol=1e-10,rtol=0):e.append(f'invalid retained schedule {x.get("backend")}')
            nx,ny=int(x['Nx']),int(x['Ny']); shapes={'V':(len(ts),),'psi_avg':(len(ts),),'T_avg':(len(ts),),'psi_real':(len(ts),ny+1,nx+1),'psi_imag':(len(ts),ny+1,nx+1),'T':(len(ts),ny+1,nx+1),'Ax':(len(ts),ny+1,nx),'Ay':(len(ts),ny,nx+1)}
            for k,s in shapes.items():
                if k not in f or tuple(f[k].shape)!=s:e.append(f'dimension mismatch {k} {x.get("backend")}')
    except (OSError,KeyError,ValueError,TypeError) as exc:return [f'corrupt output {out}: {exc}'],None
    return e,out
def plot_outputs(rp,cp,path):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    with h5py.File(rp) as r,h5py.File(cp) as c:
        fig,ax=plt.subplots(3,1,figsize=(7,7),sharex=True)
        for ds,label,axis in (('V','V',ax[0]),('T_avg','T',ax[1]),('psi_avg','mean |psi|^2',ax[2])):
            if ds in r and ds in c:
                axis.plot(r['times'][:],r[ds][:],label='reference');axis.plot(c['times'][:],c[ds][:],'--',label='candidate');axis.set_ylabel(label)
        ax[0].legend();ax[-1].set_xlabel('time (native model units)');fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);p.add_argument('--plot');a=p.parse_args();r,c=load(a.reference),load(a.candidate)
    e=meta_errors(r,c);v={}
    if not e:
        if not r.get('save_fields') or not c.get('save_fields'):e.append('accuracy comparison requires saved fields')
        else:
            er,rp=validate_artifact(a.reference,r);ec,cp=validate_artifact(a.candidate,c);e+=er+ec
            if rp is None or cp is None: e.append('missing/corrupt paired output')
            else: ee,v=field_errors(rp,cp);e+=ee
            e += [f'compatible field error exceeds 1e-11: {k}={err}' for k,err in v.items() if err>1e-11]
            if a.plot and rp is not None and cp is not None:plot_outputs(rp,cp,a.plot)
    speed=float(r['wall_seconds'])/float(c['wall_seconds']) if not e else None
    out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['backend','wall_seconds','rhs_seconds','accepted_steps','rejected_steps','rhs_calls'])
        for x in (r,c):w.writerow([x.get(k) for k in ('backend','wall_seconds','rhs_seconds','accepted_steps','rejected_steps','rhs_calls')])
    report=out.with_suffix('.md')
    with report.open('w',encoding='utf-8') as f:
        f.write('# CPU benchmark comparison\n\n- Gate: **'+('PASS' if not e else 'FAIL')+'**\n')
        f.write('- Accuracy status: **'+('fieldwise measured' if v else 'not established')+'**\n')
        if e:f.write('\nFailures:\n'+'\n'.join('- '+x for x in e)+'\n')
        else:f.write(f'\nMeasured runtime ratio (reference/candidate): **{speed:.6g}**\n')
        if v:f.write('\nNormalized field errors:\n'+'\n'.join(f'- {k}: {x:.6g}' for k,x in v.items())+'\n')
    print(json.dumps({'passed':not e,'errors':e,'field_errors':v,'speedup':speed,'csv':str(out),'report':str(report)}));return 0 if not e else 2
if __name__=='__main__':sys.exit(main())
