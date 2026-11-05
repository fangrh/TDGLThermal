import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.stats import t

sys.path.insert(0,str(Path(__file__).parents[1]/'bench'))
import analyze_ndr_state_response as ar


def write_json(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def hdf(path,voltage):
    with h5py.File(path,'w') as f:
        f['times']=np.arange(3000,5001,dtype=float)
        for key,value in [('V',voltage),('T_avg',.27),('psi_avg',.47)]: f[key]=np.full(2001,value)
        for key,value in [('psi_real',.7),('psi_imag',0.),('T',.27),('Ax',0.),('Ay',0.)]:
            f[key]=np.full((2001,3,3),value)


def metadata(config,seed,je,out):
    return dict(output_path=str(out),output_hash=ar._sha(out),Je=je,seed=seed,noise=.001,horizon=5000,
        final_time=5000,tolerance=1e-4,Nx=60,Ny=30,hx=.5,hy=.5,complete=True,status='success',
        config_hash=ar._sha(config),model_hash=ar.MODEL_HASH,rhs_source_hash=ar.RHS_HASH,
        precision='Float64',solver='Tsit5',backend='cpu_compatible',save_fields=True,
        snapshot_interval=1,skip_ratio=.6,output_schema='streamed_hdf5_fields',
        stream_maxiters_override=10000000,stream_guard_source_hash=ar.prep.GUARD_HASH)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    config=tmp_path/'config.yaml';config.write_bytes(b'fixture configuration')
    monkeypatch.setattr(ar,'CONFIG_HASH',ar._sha(config));monkeypatch.setattr(ar.prep,'CONFIG_HASH',ar._sha(config))
    monkeypatch.setattr(ar,'validate_artifact',lambda *_:([],None))
    # The finite scanner, scalar reader and conditioning byte check operate on real thin HDF5.
    baseline=tmp_path/'baseline'
    for label,je in ar.SOURCE.items():
        for seed in ar.SEEDS:
            folder=baseline/f'screen_{label}_seed{seed}'/'segment9';folder.mkdir(parents=True)
            out=folder/'source.h5';hdf(out,je*10)
            rng=folder/'rng_after.bin';rng.write_bytes(f'source RNG {label}/{seed}'.encode())
            m=metadata(config,seed,je,out);write_json(out.with_suffix('.json'),m)
            write_json(folder/'chain.json',dict(segment=9,seed=seed,Je=je,cumulative_start=45000,
                cumulative_end=50000,manifest=m,rng_after_path=str(rng),rng_after_sha256=ar._sha(rng)))
    plan=tmp_path/'plan';ar.prep.prepare(baseline,config,plan)
    jobs=ar._read(plan/'state_response_jobs.json');root=tmp_path/'campaign'
    for job in jobs:
        prev_h5=job['initial_state_path'];prev_rng=job['initial_rng_path']
        for step,je in enumerate(job['currents'],1):
            folder=root/job['id']/f'step{step}_Je{je}';folder.mkdir(parents=True)
            out=folder/'output.h5';hdf(out,je*10)
            rng=folder/'rng_after.bin';rng.write_bytes(f"{job['source_label']}/{job['seed']}/{step}".encode())
            initial=folder/'initial.bin';initial.write_bytes(f"initial {job['source_label']}/{job['seed']}/{step}".encode())
            m=metadata(config,job['seed'],je,out)
            m.update(initial_state_source=prev_h5,initial_source_hash=ar._sha(prev_h5),
                     initial_vector_path=str(initial),initial_vector_hash=ar._sha(initial),accepted_steps=100,rejected_steps=0,rhs_calls=600)
            write_json(out.with_suffix('.json'),m)
            e=dict(id=job['id'],step=step,seed=job['seed'],Je=je,cumulative_start=50000+(step-1)*5000,
                cumulative_end=50000+step*5000,preceding_hdf5=prev_h5,preceding_rng=prev_rng,
                rng_before_sha256=ar._sha(prev_rng),rng_after_path=str(rng),rng_after_sha256=ar._sha(rng),manifest=m)
            write_json(folder/'chain.json',e);prev_h5=str(out);prev_rng=str(rng)
    return plan/'protocol.json',root


def test_valid_complete_fixture_and_unvalidated_flags(fixture):
    protocol,root=fixture
    result=ar.analyze(protocol,root,True)
    assert result['errors']==[]
    assert result['complete'] and result['conditioning_identity_checked']
    assert len(result['rows'])==24 and len(result['segment_rows'])==120
    assert not result['stationary_branch_confirmed']
    unchecked=ar.analyze(protocol,root,False)
    assert unchecked['complete'] and not unchecked['conditioning_identity_checked']
    assert all(not c['negative_candidate'] for pairs in unchecked['contrasts'].values() for c in pairs.values())


@pytest.mark.parametrize('kind',['missing','je','config','rng','initial','guard','source_age','matrix','nonfinite'])
def test_tampering_disables_verified_result(fixture,kind):
    protocol,root=fixture; path=next(root.glob('*/step3_*/chain.json'));entry=ar._read(path)
    if kind=='missing': path.unlink()
    elif kind=='initial': Path(entry['manifest']['initial_vector_path']).write_bytes(b'corrupt')
    elif kind=='source_age':
        p=ar._read(protocol);s=p['sources'][0];e=ar._read(s['chain_path']);e['cumulative_end']=45000
        write_json(Path(s['chain_path']),e);s['chain_sha256']=ar._sha(s['chain_path']);write_json(protocol,p)
    elif kind=='matrix':
        p=ar._read(protocol);jobs_path=protocol.parent/p['jobs']['path'];jobs=ar._read(jobs_path);jobs[0]['currents'][2]+=.001
        write_json(jobs_path,jobs);p['jobs']['jobs_sha256']=ar._sha(jobs_path);write_json(protocol,p)
    elif kind=='nonfinite':
        with h5py.File(entry['manifest']['output_path'],'r+') as f: f['T'][0,0,0]=np.nan
        entry['manifest']['output_hash']=ar._sha(entry['manifest']['output_path']);write_json(path,entry)
    else:
        if kind=='je': entry['Je']+=.01
        elif kind=='config': entry['manifest']['config_hash']='bad'
        elif kind=='rng': entry['rng_before_sha256']='0'*64
        elif kind=='guard': entry['manifest']['stream_maxiters_override']=1000000
        write_json(path,entry)
    result=ar.analyze(protocol,root,True)
    assert not result['complete'] and result['errors']
    assert all(not c['negative_candidate'] for pairs in result['contrasts'].values() for c in pairs.values())


def test_conditioning_compares_dataset_bytes_not_container_hash(fixture):
    protocol,root=fixture
    first=next(root.glob('inner1_minus*/step1_*/chain.json'));e=ar._read(first)
    with h5py.File(e['manifest']['output_path'],'r+') as f: f['V'][0]+=1
    new_hash=ar._sha(e['manifest']['output_path']);e['manifest']['output_hash']=new_hash;write_json(first,e)
    second=next(first.parent.parent.glob('step2_*/chain.json'));later=ar._read(second)
    later['manifest']['initial_source_hash']=new_hash;write_json(second,later)
    result=ar.analyze(protocol,root,True)
    assert not result['complete']
    assert any('conditioning identity mismatch' in error for error in result['errors'])
    assert not any('output hash mismatch' in error or 'HDF5 continuity mismatch' in error for error in result['errors'])


def test_bonferroni_critical_value():
    assert ar.paired([-.02,-.021,-.019,-.022])['tcrit']==pytest.approx(float(t.ppf(1-.05/4,3)))
