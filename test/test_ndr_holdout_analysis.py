import pytest, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"bench"))
import analyze_ndr_holdout as ah

def test_paired_stats_student_lower_and_delta_upper():
    r=ah.paired_stats([.2,.22,.24,.26,.28,.3,.32,.34],[-.2,-.22,-.24,-.26,-.28,-.3,-.32,-.34])
    assert r["n"]==8 and r["nominal_one_sided_95_lower"]>.15 and r["nominal_one_sided_95_upper_delta"]<-.01

def test_wrong_manifest_is_rejected():
    e={"Je":.2,"seed":1,"manifest":{"horizon":4999,"model_hash":"wrong"}}
    errs=ah._validate(e,{"Je":.2,"seed":1},{},{})
    assert any("horizon mismatch" in x for x in errs) and any("model hash" in x for x in errs)

def test_missing_job_matrix_is_incomplete(tmp_path):
    cfg=tmp_path/"cfg.yaml";cfg.write_text("x",encoding="utf-8")
    hp={"schema":"tdgl-full-curl-ndr-holdout-v1","variants":[{"id":"candidate","config":"cfg.yaml"}],"currents":[.2,.21],"seeds":[1,2],"jobs":{"path":"jobs.json"},"source":{"screen_config_sha256":{"candidate":ah._sha(cfg)},"model_hash":ah.MODEL_HASH,"rhs_source_hash":ah.RHS_HASH},"primary_contrast":{"lo":.2,"hi":.21}}
    (tmp_path/"jobs.json").write_text("[]",encoding="utf-8");p=tmp_path/"holdout.json";p.write_text(__import__("json").dumps(hp),encoding="utf-8")
    r=ah.analyze(p,tmp_path/"campaign")
    assert not r["complete"] and r["errors"]

def test_nonfinite_fails_finite_gate():
    assert ah._finite(float("nan")) is None


def _holdout(tmp_path, monkeypatch):
    import json,hashlib
    import numpy as np
    cfg=tmp_path/"candidate.yaml";cfg.write_text("candidate",encoding="utf-8")
    currents=[.1975,.2,.2025,.205,.2075,.21,.2125]
    seeds=list(range(2026110511,2026110519)); jobs=[]; campaign=tmp_path/"campaign"
    for ji,je in enumerate(currents):
        for seed in seeds:
            jid=f"candidate_holdout_i{ji:03d}_seed{seed}"
            jobs.append({"id":jid,"config_path":str(cfg),"Je":je,"seed":seed,"segments":10,"noise":.001,"Nx":60,"Ny":30,"prior_cumulative_horizon":0,"initial_state_path":None,"initial_rng_path":None})
            previous_output=previous_rng=None
            for seg in range(1,11):
                token=lambda label:hashlib.sha256(f"{jid}/{seg}/{label}".encode()).hexdigest()
                out,after=token("out"),token("rng")
                e={"id":jid,"seed":seed,"Je":je,"segment":seg,"cumulative_start":(seg-1)*5000,"cumulative_end":seg*5000,"rng_before_sha256":previous_rng or token("initial"),"rng_after_sha256":after,"preceding_sha256":previous_output,
                   "manifest":{"Je":je,"seed":seed,"horizon":5000,"final_time":5000,"skip_ratio":.6,"noise":.001,"tolerance":1e-4,"snapshot_interval":1,"Nx":60,"Ny":30,"hx":.5,"hy":.5,"precision":"Float64","solver":"Tsit5","complete":True,"status":"success","model_hash":ah.MODEL_HASH,"rhs_source_hash":ah.RHS_HASH,"config_hash":ah._sha(cfg),"backend":"cpu_compatible","save_fields":True,"output_schema":"streamed_hdf5_fields","output_hash":out,"initial_source_hash":previous_output}}
                folder=campaign/jid/f"segment{seg}";folder.mkdir(parents=True)
                (folder/"chain.json").write_text(json.dumps(e),encoding="utf-8")
                previous_output,previous_rng=out,after
    jp=tmp_path/"holdout_jobs.json";jp.write_text(json.dumps(jobs),encoding="utf-8")
    source={"model_hash":ah.MODEL_HASH,"rhs_source_hash":ah.RHS_HASH,"screen_config_sha256":{"candidate":ah._sha(cfg),"baseline":"f"*64}}
    for name in ("screen_analysis","screen_protocol","screen_jobs"):
        path=tmp_path/(name+".json");path.write_text("{}",encoding="utf-8");source[name]=str(path);source[name+"_sha256"]=ah._sha(path)
    hp={"schema":"tdgl-full-curl-ndr-holdout-v1","selected_candidate":{"variant":"candidate","screen_bracket":[.2,.21]},"currents":currents,"seeds":seeds,"source":source,
        "jobs":{"path":jp.name,"jobs_sha256":ah._sha(jp),"count":56,"segments":10,"segment_horizon":5000,"total_horizon":50000,"snapshot_interval":1,"skip_ratio":.6,"tolerance":1e-4,"backend":"cpu_compatible","precision":"Float64","integrator":"Tsit5"},
        "primary_contrast":{"only":True,"lo":.2,"hi":.21,"final_window":[48000,50000],"prior_window":[43000,45000]}}
    pp=tmp_path/"holdout_protocol.json";pp.write_text(json.dumps(hp),encoding="utf-8")
    def data(e):
        v=[2.1,2,1.85,1.7,1.55,1.4,1.5][currents.index(e["Je"])]
        v+=.001*(e["seed"]-seeds[0])
        if e["segment"]==9:v-=.0001
        elif e["segment"]<9:v+=5 # Must not use segment 1 for drift.
        return {"times":np.arange(3000.,5001.),"V":np.full(2001,v),"T_avg":np.full(2001,.2),"psi_avg":np.full(2001,.8),"V_integrated":v}
    monkeypatch.setattr(ah,"read_segment",data)
    monkeypatch.setattr(ah,"validate_chain_artifacts",lambda _:[])
    monkeypatch.setattr(ah,"_finite_fields_errors",lambda _:[])
    return pp,campaign,jp,data


def test_actual_builder_schema_complete_and_frozen_contrast(tmp_path,monkeypatch):
    pp,campaign,_,_=_holdout(tmp_path,monkeypatch)
    r=ah.analyze(pp,campaign,True)
    assert r["complete"] and r["errors"]==[]
    assert r["primary"]["stronger_drop_pass"] and r["primary"]["stats"]["n"]==8
    assert all(x["mean"]==pytest.approx(.0001) for x in r["drift"].values())
    assert all(x["within_abs_0p0025"] for x in r["drift"].values())
    assert r["rows"][0]["psi_avg"]==pytest.approx(.8)


@pytest.mark.parametrize("kind",["rng","config","jobs","schedule","missing_seed"])
def test_integrity_failures_disable_primary_claim(tmp_path,monkeypatch,kind):
    import json
    pp,campaign,jp,original=_holdout(tmp_path,monkeypatch)
    if kind=="rng":
        f=next(campaign.glob("*/segment1/chain.json"));e=json.loads(f.read_text());e.pop("rng_before_sha256");f.write_text(json.dumps(e))
    elif kind=="config":(tmp_path/"candidate.yaml").write_text("changed")
    elif kind=="jobs":jp.write_bytes(jp.read_bytes()+b" ")
    elif kind=="schedule":
        def shifted(e):
            d=original(e);d["times"]+=1;return d
        monkeypatch.setattr(ah,"read_segment",shifted)
    else:
        jobs=json.loads(jp.read_text());jobs[-1]["seed"]=jobs[-2]["seed"];jp.write_text(json.dumps(jobs))
        hp=json.loads(pp.read_text());hp["jobs"]["jobs_sha256"]=ah._sha(jp);pp.write_text(json.dumps(hp))
    r=ah.analyze(pp,campaign,True)
    assert not r["complete"] and r["errors"] and not r["primary"]["stronger_drop_pass"]
