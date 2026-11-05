import json, sys
from pathlib import Path
import h5py
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
import analyze_ndr_enhancement_history as ah


def fixture(tmp_path):
    cfg=tmp_path/"cfg.yaml"; cfg.write_text("cfg",encoding="utf-8"); ch=ah.sha(cfg)
    currents=[.18,.19,.20,.21,.22,.23,.24]; jobs=[]; sources={}; root=tmp_path/"campaign"; root.mkdir()
    for direction in ah.DIRECTIONS:
        seq=currents if direction=="ascending" else currents[::-1]
        for seed in ah.SEEDS:
            jid=f"{direction}_seed{seed}"
            initial=tmp_path/(jid+"_initial.h5");initial.write_bytes(b"initial state")
            irng=tmp_path/(jid+"_initial.bin");irng.write_bytes(jid.encode())
            jobs.append({"id":jid,"seed":seed,"currents":seq,"config_path":str(cfg),"noise":.001,"initial_state_path":str(initial),"initial_rng_path":str(irng),"prior_cumulative_horizon":50000})
            sources[jid]=[{"id":"source_"+jid,"segment":10,"cumulative_end":50000,"seed":seed,"Je":seq[0],"manifest":{"output_path":str(initial),"output_hash":ah.sha(initial)},"rng_after_path":str(irng),"rng_after_sha256":ah.sha(irng)}]
            previous_output, previous_rng = str(initial), str(irng)
            for step,je in enumerate(seq,1):
                folder=root/jid/f"step{step}_Je{je}";folder.mkdir(parents=True)
                h5=folder/"out.h5"; times=np.arange(12000,20001,dtype=float)
                with h5py.File(h5,"w") as h:
                    h["times"]=times; h["V"]=np.full(8001,3-je*2); h["T_avg"]=np.full(8001,.27); h["psi_avg"]=np.full(8001,.45); h["Ax"]=np.zeros((8001,3,3))
                m={"output_path":str(h5),"output_hash":ah.sha(h5),"config_hash":ch,"model_hash":ah.MODEL_HASH,"rhs_source_hash":ah.RHS_HASH,"Je":je,"seed":seed,"horizon":20000,"final_time":20000,"snapshot_interval":1,"skip_ratio":.6,"tolerance":1e-4,"noise":.001,"Nx":60,"Ny":30,"hx":.5,"hy":.5,"solver":"Tsit5","precision":"Float64","save_fields":True,"output_schema":"streamed_hdf5_fields","backend":"cpu_compatible","complete":True,"status":"success","initial_source_hash":ah.sha(Path(previous_output))}
                rng=folder/"rng_after.bin"; rng.write_bytes(f"{jid}/{step}".encode())
                before=ah.sha(Path(previous_rng))
                e={"id":jid,"step":step,"Je":je,"seed":seed,"cumulative_start":50000+(step-1)*20000,"cumulative_end":50000+step*20000,"manifest":m,"rng_after_path":str(rng),"rng_after_sha256":ah.sha(rng),"rng_before_sha256":before,"preceding_hdf5":previous_output,"preceding_rng":previous_rng,"preceding_sha256":previous_output}
                (folder/"chain.json").write_text(json.dumps(e),encoding="utf-8")
                previous_output, previous_rng=str(h5), str(rng)
    jp=tmp_path/"history_jobs.json"; jp.write_text(json.dumps(jobs),encoding="utf-8")
    hp={"schema":"tdgl-full-curl-ndr-history-plan-v1","currents":currents,"seeds":list(ah.SEEDS),"directions":list(ah.DIRECTIONS),"backend":"cpu_compatible","tolerance":1e-4,"source_hashes":{"config":ch,"model":ah.MODEL_HASH,"rhs_source":ah.RHS_HASH},"jobs":{"path":jp.name,"jobs_sha256":ah.sha(jp),"segment_horizon":20000,"snapshot_interval":1,"skip_ratio":.6,"prior_cumulative_horizon":50000},"raw_window":[12000,20000],"analysis_final_window":[18000,20000],"drift_windows":[[13000,15000],[18000,20000]]}
    cp=tmp_path/"campaign.json"; cp.write_text(json.dumps({"chains":sources}), encoding="utf-8")
    hp["campaign_sha256"]=ah.sha(cp)
    hp["holdout_campaign"]=str(cp)
    hp["confidence"]={"bonferroni_contrasts":12,"stationarity_ci_abs_max":.0025}
    oldjobs=tmp_path/"holdout_jobs.json";oldjobs.write_text("[]",encoding="utf-8")
    oldp=tmp_path/"holdout_protocol.json";oldp.write_text(json.dumps({"jobs":{"path":oldjobs.name}}),encoding="utf-8")
    hp.update(holdout_protocol=str(oldp),holdout_protocol_sha256=ah.sha(oldp),holdout_jobs_sha256=ah.sha(oldjobs))
    pp=tmp_path/"history_protocol.json"; pp.write_text(json.dumps(hp),encoding="utf-8")
    return pp,root


def test_happy_history_analysis(tmp_path):
    pp,root=fixture(tmp_path); r=ah.analyze(pp,root)
    assert r["complete"] and len(r["rows"])==56
    assert set(r["deltas"])==set(ah.DIRECTIONS)
    assert r["stationary_branch_confirmed"] is False and r["ordered_negative_pair_found"] is False


def test_verified_flags_and_simultaneous_bounds(tmp_path,monkeypatch):
    pp,root=fixture(tmp_path)
    # HDF5 geometry/artifact validation itself is covered by the generic validator;
    # this fixture tests ordered receipt, schedule and inference integration.
    monkeypatch.setattr(ah,"validate_artifact",lambda *_:([],None))
    monkeypatch.setattr(ah,"_finite_fields_errors",lambda _:[])
    r=ah.analyze(pp,root,True)
    assert r["complete"] and r["ordered_negative_pair_found"]
    assert r["finite_saved_fields_validation"] and not r["stationary_branch_confirmed"]
    assert r["deltas"]["ascending"][0]["stats"]["tcrit_bonferroni12"]==pytest.approx(6.231543473366502)


def test_mismatched_order_is_rejected(tmp_path):
    pp,root=fixture(tmp_path); p=next(root.glob("ascending_seed*/step2*/chain.json")); e=json.loads(p.read_text()); e["Je"]=.24; p.write_text(json.dumps(e))
    r=ah.analyze(pp,root); assert not r["complete"] and any("current order" in x for x in r["errors"])


def test_schedule_mismatch_is_rejected(tmp_path):
    pp,root=fixture(tmp_path); p=next(root.glob("descending_seed*/step7*/out.h5"))
    with h5py.File(p,"r+") as h: h["times"][0]=123
    r=ah.analyze(pp,root); assert not r["complete"] and any("schedule" in x for x in r["errors"])


def test_rng_metadata_mismatch_is_rejected(tmp_path):
    pp,root=fixture(tmp_path); p=next(root.glob("ascending_seed*/step3*/chain.json")); e=json.loads(p.read_text()); e["rng_after_sha256"]="0"*64; p.write_text(json.dumps(e))
    r=ah.analyze(pp,root); assert not r["complete"] and any("continuity" in x or "RNG" in x for x in r["errors"])


def test_initial_rng_link_is_required(tmp_path):
    pp,root=fixture(tmp_path);p=next(root.glob("ascending_seed*/step1*/chain.json"));e=json.loads(p.read_text());e['rng_before_sha256']='a'*64;p.write_text(json.dumps(e))
    r=ah.analyze(pp,root);assert not r['complete'] and any('RNG continuity' in x for x in r['errors'])


def test_missing_initial_state_is_rejected(tmp_path):
    pp,root=fixture(tmp_path);j=json.loads((tmp_path/'history_jobs.json').read_text())[0]
    Path(j['initial_state_path']).unlink()
    r=ah.analyze(pp,root);assert not r['complete'] and any('HDF5 continuity' in x for x in r['errors'])


def test_integrated_voltage_uses_same_final_window():
    times=np.arange(12000.,20001.);ax=np.broadcast_to((-2*times/30)[:,None,None],(8001,3,3)).copy()
    ax[times<18000]+=100
    assert ah.integrated(ax,times)==pytest.approx(2)


def test_history_rng_must_match_recorded_50k_source(tmp_path):
    pp,root=fixture(tmp_path);hp=json.loads(pp.read_text());cp=Path(hp['holdout_campaign']);source=json.loads(cp.read_text())
    first=next(iter(source['chains'].values()))[0];first['rng_after_sha256']='a'*64
    cp.write_text(json.dumps(source));hp['campaign_sha256']=ah.sha(cp);pp.write_text(json.dumps(hp))
    r=ah.analyze(pp,root)
    assert not r['complete'] and any('holdout RNG lineage' in x for x in r['errors'])


def test_v2_requires_declared_guard_in_every_manifest(tmp_path):
    pp,root=fixture(tmp_path);hp=json.loads(pp.read_text());hp['schema']='tdgl-full-curl-ndr-history-plan-v2'
    hp['solver_guard']={'maxiters_override':10000000,'fail_fast':True,'source_sha256':'f'*64};pp.write_text(json.dumps(hp))
    for p in root.glob('*/step*/chain.json'):
        e=json.loads(p.read_text());e['manifest'].update(stream_maxiters_override=10000000,stream_guard_source_hash='f'*64);p.write_text(json.dumps(e))
    assert ah.analyze(pp,root)['complete']
    p=next(root.glob('*/step1*/chain.json'));e=json.loads(p.read_text());e['manifest']['stream_maxiters_override']=1000000;p.write_text(json.dumps(e))
    r=ah.analyze(pp,root)
    assert not r['complete'] and any('streaming guard metadata' in x for x in r['errors'])
