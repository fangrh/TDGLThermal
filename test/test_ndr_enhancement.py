import json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
import analyze_ndr_enhancement as ae

def test_pair_score_is_minimum_seed_fraction_not_mean():
    rows=[]
    for seed,left,right in [(1,1.0,.8),(2,1.0,.1)]:
        rows += [{"variant":"candidate","Je":.2,"seed":seed,"V":left},{"variant":"candidate","Je":.21,"seed":seed,"V":right}]
    assert ae._pair_rows(rows,[.2,.21])[0]["score"] == pytest.approx(.2)

def test_low_left_voltage_excludes_denominator():
    rows=[{"variant":"candidate","Je":.2,"seed":1,"V":.04},{"variant":"candidate","Je":.21,"seed":1,"V":0.0}]
    out=ae._pair_rows(rows,[.2,.21])[0]
    assert not out["eligible"] and out["excluded"][0]["reason"]=="low_left_voltage"

def test_nonimproving_candidate_is_not_better_than_baseline():
    rows=[]
    for variant,drop in [("baseline",.4),("candidate",.2)]:
        for seed in (1,2): rows += [{"variant":variant,"Je":.2,"seed":seed,"V":1.0},{"variant":variant,"Je":.21,"seed":seed,"V":1-drop}]
    out=ae._pair_rows(rows,[.2,.21]); scores={x["variant"]:x["score"] for x in out}
    assert scores["candidate"] < scores["baseline"]
    assert ae._select_candidate(out,scores["baseline"]) is None

def test_missing_declared_adjacent_point_cannot_create_fake_pair():
    rows=[{"variant":"candidate","Je":.2,"seed":1,"V":1.0},{"variant":"candidate","Je":.21,"seed":1,"V":.1}]
    out=ae._pair_rows(rows,[.2,.205,.21],expected_seeds=[1,2],variants=["candidate"])
    assert not any(x["eligible"] for x in out)

def test_selection_is_disabled_by_global_errors():
    outcomes=[{"variant":"candidate","Je_left":.2,"Je_right":.21,"score":.8,"eligible":True}]
    assert ae._select_candidate(outcomes,.1,errors=["missing seed"]) is None

def _receipt(je,seed,seg,model=ae.MODEL_HASH):
    return {"Je":je,"seed":seed,"segment":seg,"cumulative_start":(seg-1)*5000,"cumulative_end":seg*5000,"manifest":{"seed":seed,"Je":je,"horizon":5000,"skip_ratio":.6,"noise":.001,"tolerance":1e-4,"snapshot_interval":1,"Nx":60,"Ny":30,"hx":.5,"hy":.5,"precision":"ComplexF64/Float64","solver":"Tsit5","complete":True,"status":"success","final_time":5000,"model_hash":model,"rhs_source_hash":ae.RHS_HASH,"config_hash":"bad"}}

def test_wrong_model_hash_is_reported():
    e=_receipt(.2,1,1,model="wrong")
    errors=ae._protocol_ok(e,{"seed":1,"Je":.2,"Nx":60,"Ny":30},{}, {})
    assert any("model_hash mismatch" in x for x in errors)

def test_missing_seed_and_point_are_incomplete(tmp_path):
    cfg=tmp_path/"baseline.yaml"; cfg.write_text("x",encoding="utf-8")
    protocol={"schema":"tdgl-ndr-enhancement-screen-v1","currents":[.2,.21],"variants":[{"id":"baseline","config":"baseline.yaml"}],"seeds":[1],"jobs":{"path":"screen_jobs.json"},"config_sha256":{"baseline":ae._sha(cfg)}}
    (tmp_path/"screen_jobs.json").write_text("[]",encoding="utf-8")
    pp=tmp_path/"protocol.json"; pp.write_text(json.dumps(protocol),encoding="utf-8")
    result=ae.analyze(pp,tmp_path/"campaign")
    assert result["complete"] is False and result["errors"]

def _screen(tmp_path,monkeypatch):
    import hashlib
    import numpy as np
    variants=[];hashes={};jobs=[];campaign=tmp_path/"campaign"
    for variant in ("baseline","candidate"):
        cfg=tmp_path/"configs"/(variant+".yaml");cfg.parent.mkdir(exist_ok=True)
        cfg.write_text("configuration "+variant,encoding="utf-8")
        variants.append({"id":variant,"config":"configs/"+cfg.name});hashes[variant]=ae._sha(cfg)
        for ji,je in enumerate((.2,.21)):
            for seed in (1,2):
                jid=f"{variant}_i{ji:03d}_seed{seed}"
                job={"id":jid,"config_path":str(cfg),"Je":je,"seed":seed,"segments":2,"noise":.001}
                jobs.append(job);prev_output=None;prev_rng=None
                for segment in (1,2):
                    e=_receipt(je,seed,segment);e["id"]=jid
                    token=lambda label:hashlib.sha256(f"{jid}/{segment}/{label}".encode()).hexdigest()
                    output=token("output");after=token("rng")
                    e.update(rng_before_sha256=prev_rng or token("initial_rng"),rng_after_sha256=after,preceding_sha256=prev_output)
                    e["manifest"].update(config_hash=hashes[variant],backend="cpu_compatible",save_fields=True,
                        output_schema="streamed_hdf5_fields",output_hash=output,initial_source_hash=prev_output)
                    folder=campaign/jid/f"segment{segment}";folder.mkdir(parents=True)
                    (folder/"chain.json").write_text(json.dumps(e),encoding="utf-8")
                    prev_output,prev_rng=output,after
    jp=tmp_path/"screen_jobs.json";jp.write_text(json.dumps(jobs),encoding="utf-8")
    protocol={"schema":"tdgl-full-curl-ndr-enhancement-screen-v1","backend":"cpu_compatible",
        "currents":[.2,.21],"seeds":[1,2],"variants":variants,"config_sha256":hashes,
        "normalized_source_hashes":{"model":ae.MODEL_HASH,"rhs":ae.RHS_HASH},
        "jobs":{"path":jp.name,"jobs_sha256":ae._sha(jp)}}
    pp=tmp_path/"protocol.json";pp.write_text(json.dumps(protocol),encoding="utf-8")
    def segment_data(e):
        v=1.0 if e["Je"]==.2 else .5 if e["id"].startswith("candidate_") else .8
        return {"times":np.arange(3000.,5001.),"V":np.full(2001,v),"T_avg":np.full(2001,.2),
                "psi_avg":np.full(2001,.8),"V_integrated":v}
    monkeypatch.setattr(ae,"read_segment",segment_data)
    return pp,campaign,jp,segment_data


def test_complete_screen_selects_candidate_and_preserves_density(tmp_path,monkeypatch):
    pp,campaign,_,_=_screen(tmp_path,monkeypatch)
    r=ae.analyze(pp,campaign)
    assert r["complete"] and not r["errors"]
    assert r["selected_candidate"]["variant"]=="candidate"
    assert r["selected_candidate"]["score"]==pytest.approx(.5)
    assert r["baseline_best_score"]==pytest.approx(.2)
    assert r["rows"][0]["psi_avg"]==pytest.approx(.8)


def test_rng_conflict_disables_selection_without_heavy_hash_flag(tmp_path,monkeypatch):
    pp,campaign,_,_=_screen(tmp_path,monkeypatch)
    f=next(campaign.glob("candidate_*/segment2/chain.json"));e=json.loads(f.read_text(encoding="utf-8"))
    e["rng_before_sha256"]="f"*64;f.write_text(json.dumps(e),encoding="utf-8")
    r=ae.analyze(pp,campaign)
    assert not r["complete"] and r["selected_candidate"] is None
    assert any("RNG continuity mismatch" in x for x in r["errors"])


def test_changed_config_disables_selection(tmp_path,monkeypatch):
    pp,campaign,_,_=_screen(tmp_path,monkeypatch)
    (tmp_path/"configs/candidate.yaml").write_text("changed",encoding="utf-8")
    r=ae.analyze(pp,campaign)
    assert not r["complete"] and r["selected_candidate"] is None
    assert any("config hash mismatch" in x for x in r["errors"])


def test_missing_seed_substitution_is_detected_despite_same_job_count(tmp_path,monkeypatch):
    pp,campaign,jp,_=_screen(tmp_path,monkeypatch)
    jobs=json.loads(jp.read_text(encoding="utf-8"));jobs[-1]["seed"]=1;jobs[-1]["id"]+="_substituted"
    jp.write_text(json.dumps(jobs),encoding="utf-8")
    p=json.loads(pp.read_text(encoding="utf-8"));p["jobs"]["jobs_sha256"]=ae._sha(jp);pp.write_text(json.dumps(p),encoding="utf-8")
    r=ae.analyze(pp,campaign)
    assert not r["complete"] and r["selected_candidate"] is None
    assert any("duplicate variant/current/seed" in x for x in r["errors"])
    assert any("job matrix combinations mismatch" in x for x in r["errors"])


def test_jobs_hash_changes_are_detected(tmp_path,monkeypatch):
    pp,campaign,jp,_=_screen(tmp_path,monkeypatch)
    jp.write_bytes(jp.read_bytes()+b" ")
    r=ae.analyze(pp,campaign)
    assert not r["complete"] and r["selected_candidate"] is None
    assert "jobs_sha256 mismatch" in r["errors"]


def test_shifted_saved_schedule_is_not_accepted_by_length_alone(tmp_path,monkeypatch):
    pp,campaign,_,original=_screen(tmp_path,monkeypatch)
    def shifted(e):
        d=original(e);d["times"]+=1;return d
    monkeypatch.setattr(ae,"read_segment",shifted)
    r=ae.analyze(pp,campaign)
    assert not r["complete"] and r["selected_candidate"] is None
    assert any("retained schedule mismatch" in x for x in r["errors"])


def test_nonfinite_scalars_disable_selection(tmp_path,monkeypatch):
    import numpy as np
    pp,campaign,_,original=_screen(tmp_path,monkeypatch)
    def corrupted(e):
        d=original(e)
        if e["id"].startswith("candidate_"):d["V"][0]=np.nan
        return d
    monkeypatch.setattr(ae,"read_segment",corrupted)
    r=ae.analyze(pp,campaign)
    assert not r["complete"] and r["selected_candidate"] is None
    assert any("nonfinite scalar" in x for x in r["errors"])
