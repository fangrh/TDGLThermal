#!/usr/bin/env python3
"""Receipt-aware analysis of the frozen full-curl holdout campaign."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,platform,sys,re
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy
from scipy import stats
sys.path.insert(0,str(Path(__file__).parent))
from analyze_full_curl_ndr import read_segment,validate_chain_artifacts
from analyze_full_iv import _finite_fields_errors

MODEL_HASH="4464df90825a7afefdddbe90ae3b314724e6da951d1bc7da1ee2cf845eb2e422"
RHS_HASH="900b2a7ab277030b96454fd4563f926b0be983cc349f64d5fa07a0381ef287fe"
def _finite(x):
    if isinstance(x,float) and not math.isfinite(x): return None
    if isinstance(x,dict): return {k:_finite(v) for k,v in x.items()}
    if isinstance(x,list): return [_finite(v) for v in x]
    return x
def _sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def _read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def _m(e): return e.get("manifest",{}) if isinstance(e.get("manifest",{}),dict) else {}
def _variant(job,protocol):
    vs={v["id"]:v for v in protocol.get("variants",[])}; cp=str(job.get("config_path","")); hit=[v for v,x in vs.items() if Path(str(x.get("config",""))).name in Path(cp).parts]
    if len(hit)==1:return hit[0]
    hit=[v for v in vs if str(job.get("id","")).startswith(v+"_")]
    if len(hit)==1:return hit[0]
    selected=protocol.get("selected_candidate",{}).get("variant")
    return str(selected) if selected and (not vs or str(selected) in vs) else None
def paired_stats(fractions,deltas):
    n=len(fractions); tcrit=float(stats.t.ppf(.95,n-1)) if n>1 else float("nan")
    mean=float(np.mean(fractions)) if n else float("nan"); sem=float(np.std(fractions,ddof=1)/math.sqrt(n)) if n>1 else float("nan")
    dmean=float(np.mean(deltas)) if n else float("nan"); dsem=float(np.std(deltas,ddof=1)/math.sqrt(n)) if n>1 else float("nan")
    return {"n":n,"fractions":fractions,"mean_fraction":mean,"nominal_one_sided_95_lower":mean-tcrit*sem,"deltas":deltas,"mean_delta_V":dmean,"nominal_one_sided_95_upper_delta":dmean+tcrit*dsem,"tcrit":tcrit}
def _validate(e,j,hp,cfg):
    m=_m(e); out=[]
    def close(k,v):
        try:return math.isclose(float(m.get(k)),v,rel_tol=0,abs_tol=1e-9)
        except:return False
    for k,v in (("horizon",5000),("skip_ratio",.6),("noise",.001),("tolerance",1e-4),("snapshot_interval",1),("hx",.5),("hy",.5)):
        if not close(k,v):out.append(k+" mismatch")
    if int(m.get("Nx",-1))!=60 or int(m.get("Ny",-1))!=30:out.append("grid mismatch")
    if m.get("precision") not in ("ComplexF64/Float64","Float64") or m.get("solver")!="Tsit5":out.append("solver/precision mismatch")
    if m.get("backend")!="cpu_compatible":out.append("backend mismatch")
    if m.get("complete") is not True or m.get("status") not in ("success","Success","ReturnCode.Success"):out.append("incomplete status")
    if not close("final_time",5000):out.append("final_time mismatch")
    if m.get("model_hash")!=MODEL_HASH:out.append("model hash mismatch")
    if m.get("rhs_source_hash")!=RHS_HASH:out.append("rhs hash mismatch")
    if int(e.get("seed",-1))!=int(j.get("seed",-2)) or int(m.get("seed",-1))!=int(j.get("seed",-2)) or not math.isclose(float(e.get("Je",math.nan)),float(j.get("Je",math.nan)),rel_tol=0,abs_tol=1e-12) or not math.isclose(float(m.get("Je",math.nan)),float(j.get("Je",math.nan)),rel_tol=0,abs_tol=1e-12):out.append("job identity mismatch")
    if m.get("output_schema")!="streamed_hdf5_fields" or m.get("save_fields") is not True:out.append("output metadata mismatch")
    if not cfg or m.get("config_hash")!=cfg:out.append("config hash mismatch")
    return out
def analyze(protocol_path:Path,campaign_root:Path,validate=False):
    hp=_read(protocol_path); errors=[]; jobs_path=protocol_path.parent/hp["jobs"]["path"]; jobs=_read(jobs_path)
    variants={v["id"]:v for v in hp.get("variants",[])}
    if not variants and hp.get("selected_candidate",{}).get("variant"):
        variants={hp["selected_candidate"]["variant"]:{"id":hp["selected_candidate"]["variant"],"config":None}}
    cfg={};
    for vid,v in variants.items():
        configured=v.get("config")
        if configured: p=(protocol_path.parent/configured).resolve()
        else:
            candidates=[Path(j.get("config_path","")) for j in jobs if _variant(j,hp)==vid]
            p=candidates[0] if candidates else Path("")
        cfg[vid]=_sha(p) if p.is_file() else None
    source=hp.get("source",{}); expected=source.get("screen_config_sha256",{})
    selected=hp.get("selected_candidate",{}).get("variant")
    if selected not in variants or not expected.get(selected) or cfg.get(selected)!=expected.get(selected):
        errors.append("selected config provenance mismatch")
    declared_jobs_hash=hp.get("jobs",{}).get("jobs_sha256") or source.get("holdout_jobs_sha256")
    if not declared_jobs_hash or _sha(jobs_path)!=declared_jobs_hash:errors.append("jobs hash mismatch")
    for name in ("screen_analysis", "screen_protocol", "screen_jobs"):
        p=Path(source.get(name,"")); declared=source.get(name+"_sha256")
        if not declared or not p.is_file() or _sha(p)!=declared: errors.append(f"{name} provenance mismatch")
    if source.get("model_hash")!=MODEL_HASH or source.get("rhs_source_hash")!=RHS_HASH:errors.append("protocol source hash mismatch")
    if hp.get("schema")!="tdgl-full-curl-ndr-holdout-v1":errors.append("protocol schema mismatch")
    if len(hp.get("currents",[]))!=7 or len(hp.get("seeds",[]))!=8 or hp.get("jobs",{}).get("count")!=56 or hp.get("jobs",{}).get("segments")!=10:errors.append("frozen protocol counts mismatch")
    if hp.get("currents",[])!=sorted(set(hp.get("currents",[]))) or hp.get("seeds")!=list(range(2026110511,2026110519)):errors.append("frozen current/seed declarations mismatch")
    for key,value in {"segment_horizon":5000,"total_horizon":50000,"snapshot_interval":1,"skip_ratio":.6,"tolerance":1e-4,"backend":"cpu_compatible","precision":"Float64","integrator":"Tsit5"}.items():
        if hp.get("jobs",{}).get(key)!=value: errors.append(f"frozen protocol {key} mismatch")
    if hp.get("primary_contrast",{}).get("only") is not True:errors.append("primary contrast is not frozen-only")
    pc=hp.get("primary_contrast",{}); lo,hi=float(pc.get("lo",math.nan)),float(pc.get("hi",math.nan))
    step=(hi-lo)/4
    derived=[lo-step,lo,lo+step,lo+2*step,lo+3*step,hi,hi+step]
    if not hi>lo or len(hp.get("currents",[]))!=7 or not np.allclose(hp["currents"],derived,rtol=0,atol=1e-14) or hp.get("selected_candidate",{}).get("screen_bracket")!=[lo,hi]:errors.append("frozen primary bracket/grid mismatch")
    if pc.get("final_window")!=[48000,50000] or pc.get("prior_window")!=[43000,45000]:errors.append("frozen analysis windows mismatch")
    combos=[]; jobs_by={}
    for j in jobs:
        v=_variant(j,hp); j["variant"]=v; combos.append((v,float(j.get("Je",math.nan)),int(j.get("seed",-1))))
        path=Path(j.get("config_path",""))
        if not path.is_file() or _sha(path)!=expected.get(selected):errors.append(f"job config mismatch {j.get('id')}")
        if j.get("segments")!=10 or j.get("noise")!=.001 or j.get("Nx")!=60 or j.get("Ny")!=30 or j.get("prior_cumulative_horizon")!=0 or j.get("initial_state_path") is not None or j.get("initial_rng_path") is not None:errors.append(f"fresh job settings mismatch {j.get('id')}")
        if j.get("id") in jobs_by:errors.append("duplicate job id")
        jobs_by[j.get("id")]=j
    expected_combos={(v,float(x),int(s)) for v in variants for x in hp.get("currents",[]) for s in hp.get("seeds",[])}
    if len(jobs)!=56 or set(combos)!=expected_combos or len(set(combos))!=len(combos):errors.append("exact 56-job matrix mismatch")
    chains=defaultdict(list)
    for p in campaign_root.glob("**/segment*/chain.json"):
        try:e=_read(p);e["_chain_path"]=str(p);e["_segment_dir"]=str(p.parent);chains[str(e.get("id"))].append(e)
        except Exception as exc:errors.append(f"malformed receipt {p}: {exc}")
    rows=[]; artifact=[]
    unknown=set(chains)-set(jobs_by)
    if unknown: errors.append(f"unknown receipt jobs: {sorted(unknown)}")
    for jid,j in jobs_by.items():
        es=sorted(chains.get(jid,[]),key=lambda e:int(e.get("segment",0)))
        if len(es)!=10:errors.append(f"missing segments {jid}");continue
        for i,e in enumerate(es,1):
            if int(e.get("segment",0))!=i or not math.isclose(float(e.get("cumulative_start",math.nan)),(i-1)*5000,rel_tol=0,abs_tol=1e-8) or not math.isclose(float(e.get("cumulative_end",math.nan)),i*5000,rel_tol=0,abs_tol=1e-8):errors.append(f"continuity mismatch {jid}")
            errors.extend(f"{jid}: {x}" for x in _validate(e,j,hp,cfg.get(j.get("variant"))))
            if not re.fullmatch(r"[0-9a-fA-F]{64}",str(e.get("rng_after_sha256",""))):errors.append(f"invalid RNG after hash {jid}")
            if not re.fullmatch(r"[0-9a-fA-F]{64}",str(e.get("rng_before_sha256",""))):errors.append(f"invalid RNG before hash {jid}")
            if i>1:
                if not re.fullmatch(r"[0-9a-fA-F]{64}",str(e.get("rng_before_sha256",""))) or not re.fullmatch(r"[0-9a-fA-F]{64}",str(es[i-2].get("rng_after_sha256",""))):errors.append(f"invalid RNG hash {jid}")
                if e.get("rng_before_sha256")!=es[i-2].get("rng_after_sha256"):errors.append(f"RNG continuity mismatch {jid}")
                if e.get("preceding_sha256")!=_m(es[i-2]).get("output_hash"):errors.append(f"output continuity mismatch {jid}")
                if _m(e).get("initial_source_hash")!=_m(es[i-2]).get("output_hash"):errors.append(f"initial source continuity mismatch {jid}")
        if validate:
            ae=validate_chain_artifacts(es); artifact.extend(ae)
            if not ae: artifact.extend(_finite_fields_errors(es))
        try:
            seg=[read_segment(e) for e in es]; f=seg[-1]; p=seg[-2]
            if any(len(x["times"])!=2001 or not np.allclose(x["times"],np.arange(3000.,5001.),rtol=0,atol=1e-10) for x in seg):errors.append(f"retained schedule mismatch {jid}")
            def meanwin(x,lo,hi,key="V"):
                mask=(x["times"]>=lo)&(x["times"]<=hi)
                if not np.any(mask):raise ValueError("empty time window")
                return float(np.mean(x[key][mask]))
            vf=meanwin(f,3000,5000); vp=meanwin(p,3000,5000); vals=[vf,meanwin(f,3000,5000,"T_avg"),meanwin(f,3000,5000,"psi_avg"),float(f["V_integrated"]),vf-vp]
            if not all(math.isfinite(x) for x in vals):errors.append(f"nonfinite scalar {jid}")
            rows.append({"variant":j["variant"],"Je":float(j["Je"]),"seed":int(j["seed"]),"V":vf,"T_avg":vals[1],"psi_avg":vals[2],"V_integrated":vals[3],"drift":vals[4],"job_id":jid})
        except Exception as exc:errors.append(f"read {jid}: {exc}")
    errors.extend(artifact); lo=float(hp["primary_contrast"]["lo"]); hi=float(hp["primary_contrast"]["hi"]); primary=[r for r in rows if math.isclose(r["Je"],lo,abs_tol=1e-12) or math.isclose(r["Je"],hi,abs_tol=1e-12)]
    fractions=[];deltas=[]; seedrows=[]
    for s in hp.get("seeds",[]):
        a=next((r for r in primary if r["seed"]==s and math.isclose(r["Je"],lo,abs_tol=1e-12)),None); b=next((r for r in primary if r["seed"]==s and math.isclose(r["Je"],hi,abs_tol=1e-12)),None)
        if a and b and a["V"]>=.05: fractions.append((a["V"]-b["V"])/a["V"]);deltas.append(b["V"]-a["V"]);seedrows.append({"seed":s,"V_lo":a["V"],"V_hi":b["V"],"fraction":fractions[-1],"delta_V":deltas[-1]})
    ps=paired_stats(fractions,deltas); complete=not errors and len(rows)==56 and len(seedrows)==8
    drift={}
    for j in sorted(set(r["Je"] for r in rows)):
        vals=[r["drift"] for r in rows if math.isclose(r["Je"],j,abs_tol=1e-12)]; n=len(vals); tc=float(stats.t.ppf(.975,n-1)) if n>1 else float("nan"); mean=float(np.mean(vals)) if vals else float("nan"); sem=float(np.std(vals,ddof=1)/math.sqrt(n)) if n>1 else float("nan"); lo_ci,hi_ci=mean-tc*sem,mean+tc*sem
        drift[str(j)]={"n":n,"mean":mean,"ci95_low":lo_ci,"ci95_high":hi_ci,"within_abs_0p0025":n==8 and lo_ci>=-.0025 and hi_ci<=.0025}
    gates_enabled=bool(validate and complete)
    source_files=[Path(__file__),Path(__file__).with_name("analyze_full_curl_ndr.py"),Path(__file__).with_name("analyze_full_iv.py"),Path(__file__).with_name("compare_results.py")]
    result={"schema":"tdgl-ndr-holdout-analysis-v1","complete":complete,"pending":not complete,"errors":errors,"artifact_validation":validate,"finite_saved_fields_validation":validate,"protocol":hp,"provenance":{"protocol_path":str(protocol_path.resolve()),"campaign_root":str(campaign_root.resolve()),"protocol_sha256":_sha(protocol_path),"jobs_sha256":_sha(jobs_path),"config_sha256":cfg,"analysis_source_sha256":{p.name:_sha(p) for p in source_files},"model_hash":MODEL_HASH,"rhs_source_hash":RHS_HASH,"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},"rows":rows,"primary":{"lo":lo,"hi":hi,"seed_rows":seedrows,"stats":ps,"gates_enabled":gates_enabled,"gates":{"mean_fraction":gates_enabled and ps["mean_fraction"]>=.2,"lower_fraction":gates_enabled and ps["nominal_one_sided_95_lower"]>.15,"upper_delta":gates_enabled and ps["nominal_one_sided_95_upper_delta"]<-.01}},"drift":drift,"claim":"frozen finite-time holdout contrast; no stationary-NDR claim","frame_correlation_note":"Independent seed trajectories are replicates. Student bounds are nominal and can misrepresent mixed switching distributions."}
    result["primary"]["stronger_drop_pass"]=all(result["primary"]["gates"].values())
    return _finite(result)
def write_outputs(r,out):
    out.mkdir(parents=True,exist_ok=True);(out/"holdout_analysis.json").write_text(json.dumps(r,indent=2,allow_nan=False),encoding="utf-8")
    with (out/"holdout_seed_rows.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["variant","Je","seed","V","T_avg","psi_avg","V_integrated","drift","job_id"]);w.writerows([[x.get(k) for k in ("variant","Je","seed","V","T_avg","psi_avg","V_integrated","drift","job_id")] for x in r["rows"]])
    try:
        import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True); rows=r["rows"]; currents=r["protocol"]["currents"]; seeds=r["protocol"]["seeds"]; by={(x["Je"],x["seed"]):x for x in rows}; labels={"V":"V","T_avg":"T_avg","psi_avg":"mean |ψ|²"}
        for k,a in zip(("V","T_avg","psi_avg"),ax):
            for s in seeds:a.plot(currents,[by.get((j,s),{}).get(k,np.nan) for j in currents],lw=.7,alpha=.35,color="0.55")
            means=[];half=[]
            for j in currents:
                vals=[by[(j,s)][k] for s in seeds if (j,s) in by]
                means.append(float(np.mean(vals)) if len(vals)==8 else np.nan)
                half.append(float(stats.t.ppf(.975,7)*np.std(vals,ddof=1)/math.sqrt(8)) if len(vals)==8 else np.nan)
            a.errorbar(currents,means,yerr=half,fmt="o-",lw=2,capsize=3,color="#075985",label="Mean ± nominal 95% CI; 8 fresh seeds")
            a.axvspan(r["primary"]["lo"],r["primary"]["hi"],color="#f59e0b",alpha=.12,label="Frozen primary bracket")
            a.set(xlabel="Je (native model units)",ylabel=labels[k],title=f"Final 48–50kτ · {labels[k]}");a.grid(alpha=.2);a.legend(fontsize=7)
        status="Verified" if r["complete"] and r["artifact_validation"] else "PENDING"
        fig.suptitle(f"{status} full-curl holdout · {r['protocol']['selected_candidate']['variant']} · finite-time contrast",fontsize=12);fig.savefig(out/"holdout.png",dpi=180);fig.savefig(out/"holdout.pdf");plt.close(fig)
    except ImportError: pass
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--protocol",required=True,type=Path);p.add_argument("--campaign-root",required=True,type=Path);p.add_argument("--out",required=True,type=Path);p.add_argument("--validate-artifacts",action="store_true");a=p.parse_args(argv);r=analyze(a.protocol,a.campaign_root,a.validate_artifacts);write_outputs(r,a.out);print(json.dumps({"complete":r["complete"],"out":str(a.out)}));return 0 if r["complete"] else 1
if __name__=="__main__":raise SystemExit(main())
