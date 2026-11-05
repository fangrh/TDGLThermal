#!/usr/bin/env python3
"""Bounded two-segment full-curl NDR enhancement screen."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))
from analyze_full_curl_ndr import read_segment, validate_chain_artifacts
from analyze_full_iv import _finite_fields_errors

MODEL_HASH = "4464df90825a7afefdddbe90ae3b314724e6da951d1bc7da1ee2cf845eb2e422"
RHS_HASH = "900b2a7ab277030b96454fd4563f926b0be983cc349f64d5fa07a0381ef287fe"

def _finite(x):
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, dict): return {k: _finite(v) for k,v in x.items()}
    if isinstance(x, list): return [_finite(v) for v in x]
    return x
def _sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def _manifest(e): return e.get("manifest", {}) if isinstance(e.get("manifest", {}),dict) else {}
def _read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def _canonical_variant(job, variants):
    parts=Path(str(job.get("config_path",""))).parts
    matches=[v for v,x in variants.items() if (suffix:=Path(str(x.get("config",""))).parts)
             and len(parts)>=len(suffix) and parts[-len(suffix):]==suffix]
    if len(matches)==1:
        return matches[0] if job.get("variant",matches[0])==matches[0] else None
    jid=str(job.get("id",""))
    matches=[v for v in variants if jid.startswith(v+"_")]
    return matches[0] if len(matches)==1 else None

def _protocol_ok(e, job, protocol, config_hashes):
    m=_manifest(e); errors=[]
    def close(k,v):
        try: return math.isclose(float(m.get(k)),v,rel_tol=0,abs_tol=1e-9)
        except (TypeError,ValueError): return False
    for k,v in (("horizon",5000),("skip_ratio",.6),("noise",.001),("tolerance",1e-4),("snapshot_interval",1)):
        if not close(k,v): errors.append(f"{k} mismatch")
    if int(m.get("Nx",-1))!=int(job.get("Nx",60)) or int(m.get("Ny",-1))!=int(job.get("Ny",30)): errors.append("grid mismatch")
    for k,v in (("hx",.5),("hy",.5)):
        if not close(k,v): errors.append(f"{k} mismatch")
    if m.get("precision") not in ("ComplexF64/Float64","Float64"): errors.append("precision mismatch")
    if m.get("solver")!="Tsit5": errors.append("solver mismatch")
    if m.get("complete") is not True or m.get("status") not in ("success","Success","ReturnCode.Success"): errors.append("incomplete status")
    if not math.isclose(float(m.get("final_time",math.nan)),5000,rel_tol=0,abs_tol=1e-8): errors.append("final_time mismatch")
    for field, expected in (("model_hash",MODEL_HASH),("rhs_source_hash",RHS_HASH)):
        if m.get(field)!=expected: errors.append(f"{field} mismatch")
    cfg=config_hashes.get(job.get("variant"))
    if not cfg or m.get("config_hash")!=cfg: errors.append("config_hash mismatch")
    if int(e.get("seed",-1))!=int(job.get("seed",-2)) or int(m.get("seed",-1))!=int(job.get("seed",-2)) or not math.isclose(float(e.get("Je",math.nan)),float(job.get("Je",math.nan)),rel_tol=0,abs_tol=1e-12) or not math.isclose(float(m.get("Je",math.nan)),float(job.get("Je",math.nan)),rel_tol=0,abs_tol=1e-12): errors.append("job identity mismatch")
    if m.get("output_schema")!="streamed_hdf5_fields" or m.get("save_fields") is not True: errors.append("field output metadata mismatch")
    if m.get("backend")!=protocol.get("backend","cpu_compatible"): errors.append("backend mismatch")
    return errors

def _load_jobs(protocol, protocol_path):
    jobs_path=Path(protocol.get("jobs",{}).get("path","screen_jobs.json"))
    if not jobs_path.is_absolute(): jobs_path=protocol_path.parent/jobs_path
    return _read(jobs_path), jobs_path

def _pair_rows(rows, currents, expected_seeds=None, variants=None):
    by=defaultdict(list)
    for r in rows: by[(r["variant"],float(r["Je"]))].append(r)
    outcomes=[]
    for variant in sorted(variants if variants is not None else {r["variant"] for r in rows}):
        js=sorted(float(x) for x in currents)
        for left,right in zip(js,js[1:]):
            l={r["seed"]:r for r in by[(variant,left)]}; rr={r["seed"]:r for r in by[(variant,right)]}; seeds=sorted(expected_seeds if expected_seeds is not None else set(l)|set(rr))
            usable=[]; excluded=[]
            for seed in seeds:
                if seed not in l or seed not in rr: excluded.append({"seed":seed,"reason":"missing_pair"}); continue
                vl,vr=l[seed]["V"],rr[seed]["V"]
                if vl < .05: excluded.append({"seed":seed,"reason":"low_left_voltage","V_left":vl}); continue
                drop=vl-vr; frac=drop/vl
                usable.append({"seed":seed,"V_left":vl,"V_right":vr,"absolute_drop":drop,"fraction":frac,"eligible":drop>.01})
            eligible=[x for x in usable if x["eligible"]]
            outcomes.append({"variant":variant,"Je_left":left,"Je_right":right,"seeds":usable,"excluded":excluded,
                             "eligible":len(eligible)==len(seeds) and len(seeds)>0 and len(usable)==len(seeds),"score":min((x["fraction"] for x in eligible),default=None)})
    return outcomes

def _select_candidate(outcomes, baseline_score, errors=()):
    if errors: return None
    best={}
    for x in outcomes:
        if x["eligible"] and (x["variant"] not in best or x["score"]>best[x["variant"]]["score"]): best[x["variant"]]=x
    candidates=[x for v,x in best.items() if v!="baseline" and x["score"]>baseline_score]
    return max(candidates,key=lambda x:(x["score"],-x["Je_left"])) if candidates else None

def analyze(protocol_path:Path,campaign_root:Path,validate=False):
    protocol=_read(protocol_path); errors=[]
    if protocol.get("schema")!="tdgl-full-curl-ndr-enhancement-screen-v1": errors.append("protocol schema mismatch")
    currents=protocol.get("currents",[]); seeds=protocol.get("seeds",[])
    if len(currents)<2 or currents!=sorted(set(currents)): errors.append("currents must be unique and increasing")
    if len(seeds)!=2 or len(set(seeds))!=2: errors.append("screen requires two distinct declared seeds")
    try: jobs,jobs_path=_load_jobs(protocol,protocol_path)
    except Exception as exc: return {"schema":"tdgl-ndr-enhancement-analysis-v1","complete":False,"pending":True,"errors":[str(exc)]}
    variants={v["id"]:v for v in protocol.get("variants",[])}; jobs_by={j.get("id"):j for j in jobs}
    if "baseline" not in variants or len(variants)!=len(protocol.get("variants",[])): errors.append("invalid variant declarations")
    if len(jobs_by)!=len(jobs): errors.append("duplicate job id")
    expected_jobs=len(variants)*len(protocol.get("currents",[]))*len(protocol.get("seeds",[]))
    if len(jobs_by)!=expected_jobs: errors.append(f"job matrix incomplete: {len(jobs_by)}/{expected_jobs}")
    expected_combos={(v,float(j),int(s)) for v in variants for j in protocol.get("currents",[]) for s in protocol.get("seeds",[])}
    actual_combos=[]
    for j in jobs:
        v=_canonical_variant(j,variants); j["variant"]=v
        actual_combos.append((v,float(j.get("Je",math.nan)),int(j.get("seed",-1))))
        if j.get("segments")!=2 or j.get("noise")!=.001 or j.get("Nx",60)!=60 or j.get("Ny",30)!=30:
            errors.append(f"job settings mismatch {j.get('id')}")
    if len(set(actual_combos))!=len(actual_combos): errors.append("duplicate variant/current/seed combination")
    if set(actual_combos)!=expected_combos: errors.append("job matrix combinations mismatch")
    cfg_hashes={}
    for vid,v in variants.items():
        cp=protocol_path.parent/v.get("config","")
        if cp.is_file(): cfg_hashes[vid]=_sha(cp)
    expected_cfg=protocol.get("config_sha256",{});
    if set(expected_cfg)!=set(variants): errors.append("declared config hash coverage mismatch")
    for vid in variants:
        h=expected_cfg.get(vid)
        if not h or cfg_hashes.get(vid)!=h: errors.append(f"catalog config hash mismatch {vid}")
    declared_jobs_hash=protocol.get("jobs_sha256") or protocol.get("jobs",{}).get("jobs_sha256")
    if not declared_jobs_hash or _sha(jobs_path)!=declared_jobs_hash: errors.append("jobs_sha256 mismatch")
    source_hashes=protocol.get("normalized_source_hashes",{})
    if source_hashes.get("model")!=MODEL_HASH: errors.append("declared model source hash mismatch")
    if source_hashes.get("rhs")!=RHS_HASH: errors.append("declared RHS source hash mismatch")
    chains=defaultdict(list)
    for p in campaign_root.glob("**/segment*/chain.json"):
        try:
            e=_read(p); e["_chain_path"]=str(p); e["_segment_dir"]=str(p.parent); chains[str(e.get("id"))].append(e)
            if str(e.get("id")) not in jobs_by: errors.append(f"undeclared receipt job {e.get('id')}")
        except Exception as exc: errors.append(f"malformed receipt {p}: {exc}")
    rows=[]; artifact_errors=[]
    for jid,j in jobs_by.items():
        es=sorted(chains.get(jid,[]),key=lambda x:int(x.get("segment",0)))
        if len(es)!=int(j.get("segments",2)): errors.append(f"missing segments {jid}"); continue
        for i,e in enumerate(es,1):
            if int(e.get("segment",0))!=i or not math.isclose(float(e.get("cumulative_start",math.nan)),(i-1)*5000,rel_tol=0,abs_tol=1e-8) or not math.isclose(float(e.get("cumulative_end",math.nan)),i*5000,rel_tol=0,abs_tol=1e-8): errors.append(f"continuity mismatch {jid} segment {i}")
            errors.extend(f"{jid}: {x}" for x in _protocol_ok(e,j,protocol,cfg_hashes))
            for name in ("rng_before_sha256","rng_after_sha256"):
                value=e.get(name,"")
                if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
                    errors.append(f"invalid {name} {jid} segment {i}")
            if i>1:
                prev=es[i-2]; pm=_manifest(prev)
                if e.get("rng_before_sha256")!=prev.get("rng_after_sha256"): errors.append(f"RNG continuity mismatch {jid}")
                if e.get("preceding_sha256")!=pm.get("output_hash"): errors.append(f"output continuity mismatch {jid}")
                if e.get("manifest",{}).get("initial_source_hash")!=pm.get("output_hash"): errors.append(f"initial source continuity mismatch {jid}")
        if validate:
            ae=validate_chain_artifacts(es)
            artifact_errors.extend(ae)
            if not ae: artifact_errors.extend(_finite_fields_errors(es))
        try:
            s=[read_segment(e) for e in es]; final=s[-1]; prior=s[0]
            expected_times=np.arange(3000.,5001.)
            if any(len(seg["times"])!=2001 or not np.allclose(seg["times"],expected_times,rtol=0,atol=1e-10) for seg in s):
                errors.append(f"retained schedule mismatch {jid}")
            def win(seg,lo,hi,key="V"):
                mask=(seg["times"]>=lo)&(seg["times"]<=hi)
                if not np.any(mask): raise ValueError("empty time window")
                return float(np.mean(seg[key][mask]))
            vf=win(final,3000,5000); vp=win(prior,3000,5000)
            vals=[vf,win(final,3000,5000,"T_avg"),win(final,3000,5000,"psi_avg"),float(final["V_integrated"]),vf-vp]
            if not all(math.isfinite(x) for x in vals): errors.append(f"nonfinite scalar {jid}")
            rows.append({"variant":j["variant"],"Je":float(j["Je"]),"seed":int(j["seed"]),"V":vf,"T_avg":vals[1],"psi_avg":vals[2],"V_integrated":vals[3],"drift":vals[4],"job_id":jid})
        except Exception as exc: errors.append(f"read {jid}: {exc}")
    errors.extend(artifact_errors)
    outcomes=_pair_rows(rows,protocol.get("currents",[]),protocol.get("seeds",[]),variants); baseline=[x for x in outcomes if x["variant"]=="baseline" and x["eligible"]]; baseline_score=max((x["score"] for x in baseline),default=0.0)
    best={}
    for x in outcomes:
        if x["eligible"] and (x["variant"] not in best or x["score"]>best[x["variant"]]["score"]): best[x["variant"]]=x
    selected=_select_candidate(outcomes,baseline_score,errors)
    sources=[Path(__file__),Path(__file__).with_name("analyze_full_curl_ndr.py"),Path(__file__).with_name("analyze_full_iv.py"),Path(__file__).with_name("compare_results.py")]
    complete=not errors and len(rows)==len(jobs)
    return _finite({"schema":"tdgl-ndr-enhancement-analysis-v1","complete":complete,"pending":not complete,
        "artifact_validation":validate,"finite_saved_fields_validation":validate,"protocol":protocol,
        "provenance":{"protocol_path":str(protocol_path.resolve()),"protocol_sha256":_sha(protocol_path),
            "jobs_path":str(jobs_path.resolve()),"jobs_sha256":_sha(jobs_path),
            "campaign_root":str(campaign_root.resolve()),"config_sha256":cfg_hashes,
            "analysis_source_sha256":{p.name:_sha(p) for p in sources},"python":sys.version,"numpy":np.__version__},
        "errors":errors,"rows":rows,"pair_outcomes":outcomes,"variant_best":best,
        "baseline_best_score":baseline_score,"selected_candidate":selected,
        "classification":"exploratory screening only; no stationarity claim",
        "frame_correlation_note":"frames within a trajectory are not independent replicates"})

def write_outputs(result,out:Path):
    out.mkdir(parents=True,exist_ok=True); (out/"enhancement_analysis.json").write_text(json.dumps(_finite(result),indent=2,allow_nan=False),encoding="utf-8")
    with (out/"enhancement_seed_rows.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["variant","Je","seed","V","T_avg","psi_avg","V_integrated","drift","job_id"]); w.writerows([[r.get(k) for k in ("variant","Je","seed","V","T_avg","psi_avg","V_integrated","drift","job_id")] for r in result.get("rows",[])])
    with (out/"enhancement_pairs.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["variant","Je_left","Je_right","eligible","score"]); w.writerows([[x[k] for k in ("variant","Je_left","Je_right","eligible","score")] for x in result.get("pair_outcomes",[])])
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
        fig.suptitle(f"Full-curl exploratory screen · 2 seeds · {len(result.get('rows',[]))} complete chains · "+("verified coverage" if result.get("complete") else "PENDING"))
        currents=sorted(float(x) for x in result.get("protocol",{}).get("currents",[])); seeds=result.get("protocol",{}).get("seeds",[])
        for vi,variant in enumerate(result.get("protocol",{}).get("variants",[])):
            v=variant["id"]; color=plt.get_cmap("tab10")(vi); label=variant.get("changes",v)
            if v=="baseline": label="Baseline"
            rr=[r for r in result.get("rows",[]) if r["variant"]==v]; by={(r["Je"],r["seed"]):r for r in rr}
            for seed in seeds:
                ax[0].plot(currents,[by.get((j,seed),{}).get("V",np.nan) for j in currents],"-",lw=.7,alpha=.35,color=color)
            for axis,key in zip(ax,("V","T_avg","psi_avg")):
                mean=[np.mean([by[(j,s)][key] for s in seeds])
                      if all((j,s) in by and by[(j,s)].get(key) is not None and math.isfinite(by[(j,s)][key]) for s in seeds)
                      else np.nan for j in currents]
                axis.plot(currents,mean,"o-",ms=3,lw=2,label=label,color=color)
        ax[0].set(xlabel="Je (native model units)",ylabel="V",title="Full-curl screening · final 8–10kτ"); ax[1].set(xlabel="Je",ylabel="T_avg",title="Final temperature"); ax[2].set(xlabel="Je",ylabel="mean |ψ|²",title="Final order parameter"); ax[0].legend(fontsize=7); ax[1].legend(fontsize=7); ax[2].legend(fontsize=7)
        for a in ax:a.grid(alpha=.2)
        fig.savefig(out/"enhancement_screen.png",dpi=180); fig.savefig(out/"enhancement_screen.pdf"); plt.close(fig)
    except ImportError: pass

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--protocol",required=True,type=Path); p.add_argument("--campaign-root",required=True,type=Path); p.add_argument("--out",required=True,type=Path); p.add_argument("--validate-artifacts",action="store_true"); a=p.parse_args(argv); r=analyze(a.protocol,a.campaign_root,a.validate_artifacts); write_outputs(r,a.out); print(json.dumps({"complete":r["complete"],"out":str(a.out)})); return 0 if r["complete"] else 1
if __name__=="__main__": raise SystemExit(main())
