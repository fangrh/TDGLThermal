#!/usr/bin/env python3
"""Verified analysis of the 20k-tau ordered-history holdout plan."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys, re
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import scipy
from scipy.stats import t
sys.path.insert(0,str(Path(__file__).parent))
from compare_results import validate_artifact
from analyze_full_iv import _finite_fields_errors

MODEL_HASH="4464df90825a7afefdddbe90ae3b314724e6da951d1bc7da1ee2cf845eb2e422"
RHS_HASH="900b2a7ab277030b96454fd4563f926b0be983cc349f64d5fa07a0381ef287fe"
SEEDS=tuple(range(2026110511,2026110515)); DIRECTIONS=("ascending","descending")

def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def finite_json(x):
 if isinstance(x,float) and not math.isfinite(x):return None
 if isinstance(x,dict):return {k:finite_json(v) for k,v in x.items()}
 if isinstance(x,list):return [finite_json(v) for v in x]
 return x
def meanwin(ti,x,lo,hi):
 m=(ti>=lo)&(ti<=hi); return float(np.mean(x[m])) if np.any(m) else float("nan")
def integrated(ax,ti,lo=18000,hi=20000):
 m=(ti>=lo)&(ti<=hi); a=np.asarray(ax)[m]
 if len(a)<2 or hi==lo:return float("nan")
 return float(-30.0*(np.mean(a[-1,1:-1,1:-1])-np.mean(a[0,1:-1,1:-1]))/(ti[m][-1]-ti[m][0]))
def load_segment(e):
 p=Path(e["manifest"]["output_path"])
 if not p.is_file(): raise ValueError(f"missing HDF5 {p}")
 with h5py.File(p,"r") as h:
  out={k:np.asarray(h[k],dtype=float) for k in ("times","V","T_avg","psi_avg","Ax")}
 return out
def stat(values,negative=False):
 n=len(values); mean=float(np.mean(values)) if n else float("nan"); sem=float(np.std(values,ddof=1)/math.sqrt(n)) if n>1 else float("nan")
 crit=float(t.ppf(1-.05/12,n-1)) if n>1 else float("nan"); upper=mean+crit*sem if math.isfinite(sem) else float("nan")
 return {"n":n,"mean":mean,"sem":sem,"tcrit_bonferroni12":crit,"upper95_bonferroni12":upper,"values":values,"negative":bool(n==4 and negative and upper<-.01)}
def analyze(protocol:Path,campaign:Path,validate=False):
 hp=read(protocol); errors=[]
 if hp.get("schema")!="tdgl-full-curl-ndr-history-plan-v1": errors.append("protocol schema mismatch")
 jobs=read(protocol.parent/hp["jobs"]["path"]); expected={(j["id"],int(j["seed"])) for j in jobs}
 if hp.get("jobs",{}).get("jobs_sha256")!=sha(protocol.parent/hp["jobs"]["path"]): errors.append("history jobs hash mismatch")
 if len(jobs)!=8 or expected!={(f"{d}_seed{s}",s) for d in DIRECTIONS for s in SEEDS}: errors.append("history job matrix mismatch")
 if hp.get("source_hashes",{}).get("model")!=MODEL_HASH or hp.get("source_hashes",{}).get("rhs_source")!=RHS_HASH: errors.append("source hash mismatch")
 declared=[float(x) for x in hp.get("currents",[])]
 if len(declared)!=7 or declared!=sorted(set(declared)) or hp.get("seeds")!=list(SEEDS) or hp.get("directions")!=list(DIRECTIONS):errors.append("declared history grid/seeds mismatch")
 if hp.get("jobs",{}).get("segment_horizon")!=20000 or hp.get("jobs",{}).get("snapshot_interval")!=1 or hp.get("jobs",{}).get("skip_ratio")!=.6 or hp.get("jobs",{}).get("prior_cumulative_horizon")!=50000:errors.append("frozen history timing mismatch")
 if hp.get("confidence")!={"bonferroni_contrasts":12,"stationarity_ci_abs_max":.0025}:errors.append("frozen confidence rules mismatch")
 if hp.get("raw_window")!=[12000,20000] or hp.get("analysis_final_window")!=[18000,20000] or hp.get("drift_windows")!=[[13000,15000],[18000,20000]]:errors.append("frozen history windows mismatch")
 if hp.get("backend")!="cpu_compatible" or float(hp.get("tolerance",math.nan))!=1e-4: errors.append("solver protocol mismatch")
 holdout_protocol=Path(hp.get("holdout_protocol",""))
 if not holdout_protocol.is_file() or not hp.get("holdout_protocol_sha256") or hp.get("holdout_protocol_sha256")!=sha(holdout_protocol):errors.append("holdout protocol hash mismatch")
 else:
  old=read(holdout_protocol); oldjobs=holdout_protocol.parent/old.get("jobs",{}).get("path","")
  if not oldjobs.is_file() or not hp.get("holdout_jobs_sha256") or hp["holdout_jobs_sha256"]!=sha(oldjobs):errors.append("holdout jobs hash mismatch")
 campaign_source=Path(hp.get("holdout_campaign", ""))
 if not campaign_source.is_file() or not hp.get("campaign_sha256") or hp.get("campaign_sha256")!=sha(campaign_source): errors.append("campaign hash mismatch")
 grouped=defaultdict(list)
 for p in campaign.glob("**/step*/chain.json"):
  try:
   e=read(p); e["_chain_path"]=str(p); e["_segment_dir"]=str(p.parent); grouped[str(e.get("id"))].append(e)
  except Exception as exc: errors.append(f"malformed receipt {p}: {exc}")
 if set(grouped)!= {j["id"] for j in jobs}: errors.append("campaign job IDs mismatch")
 rows=[]; bydir=defaultdict(lambda:defaultdict(list))
 for job in jobs:
  jid=job["id"]; es=sorted(grouped.get(jid,[]),key=lambda e:int(e.get("step",0)))
  if len(es)!=7 or [int(e.get("step",-1)) for e in es]!=list(range(1,8)): errors.append(f"incomplete history {jid}"); continue
  cfg=Path(job.get("config_path",""))
  if not cfg.is_file() or sha(cfg)!=hp["source_hashes"]["config"]:errors.append(f"job config mismatch {jid}")
  seq=list(job["currents"]); expected_seq=declared if jid.startswith("ascending") else list(reversed(declared))
  if seq!=expected_seq:errors.append(f"declared current order mismatch {jid}")
  if [float(e.get("Je",math.nan)) for e in es] != expected_seq: errors.append(f"current order mismatch {jid}")
  for i,e in enumerate(es,1):
   m=e.get("manifest",{}); ti=None
   checks=(("horizon",20000),("final_time",20000),("snapshot_interval",1),("skip_ratio",.6),("tolerance",1e-4),("noise",.001),("Nx",60),("Ny",30))
   if any(not math.isclose(float(m.get(k,math.nan)),v,rel_tol=0,abs_tol=1e-9) for k,v in checks) or m.get("complete") is not True or m.get("status") not in ("success","Success","ReturnCode.Success") or m.get("model_hash")!=MODEL_HASH or m.get("rhs_source_hash")!=RHS_HASH or m.get("config_hash")!=hp["source_hashes"]["config"] or m.get("backend")!="cpu_compatible" or m.get("solver")!="Tsit5" or m.get("precision") not in ("Float64","ComplexF64/Float64") or m.get("hx")!=.5 or m.get("hy")!=.5 or m.get("save_fields") is not True or m.get("output_schema")!="streamed_hdf5_fields" or int(m.get("seed",-1))!=int(job["seed"]) or not math.isclose(float(m.get("Je",math.nan)),float(expected_seq[i-1]),rel_tol=0,abs_tol=1e-12): errors.append(f"metadata mismatch {jid}/{i}")
   if int(e.get("seed",-1))!=int(job["seed"]) or not math.isclose(float(e.get("Je",math.nan)),float(expected_seq[i-1]),abs_tol=1e-12): errors.append(f"identity mismatch {jid}/{i}")
   if not math.isclose(float(e.get("cumulative_start",math.nan)),50000+(i-1)*20000,rel_tol=0,abs_tol=1e-8) or not math.isclose(float(e.get("cumulative_end",math.nan)),50000+i*20000,rel_tol=0,abs_tol=1e-8): errors.append(f"cumulative mismatch {jid}/{i}")
   prev_h5=Path(job.get("initial_state_path","") if i==1 else es[i-2].get("manifest",{}).get("output_path",""))
   prev_rng=Path(job.get("initial_rng_path","") if i==1 else es[i-2].get("rng_after_path",""))
   if Path(e.get("preceding_hdf5") or "").resolve()!=prev_h5.resolve() or not prev_h5.is_file() or sha(prev_h5)!=m.get("initial_source_hash"): errors.append(f"HDF5 continuity mismatch {jid}/{i}")
   if Path(e.get("preceding_rng") or "").resolve()!=prev_rng.resolve() or not prev_rng.is_file() or sha(prev_rng)!=e.get("rng_before_sha256"): errors.append(f"RNG continuity mismatch {jid}/{i}")
   rng=Path(e.get("rng_after_path", ""))
   outp=Path(m.get("output_path",""))
   if not outp.is_file() or sha(outp)!=m.get("output_hash"): errors.append(f"output hash mismatch {jid}/{i}")
   if not rng.is_file() or e.get("rng_after_sha256")!=sha(rng): errors.append(f"RNG hash mismatch {jid}/{i}")
   if i>1 and e.get("rng_before_sha256")!=es[i-2].get("rng_after_sha256"): errors.append(f"RNG continuity mismatch {jid}/{i}")
   if any(re.fullmatch(r"[0-9a-f]{64}",str(e.get(k,""))) is None for k in ("rng_before_sha256","rng_after_sha256")):errors.append(f"invalid RNG hash {jid}/{i}")
   if validate:
    side=Path(str(m.get("output_path", "")).replace(".h5", ".json"))
    try:
     ae,_=validate_artifact(side,m); errors.extend(f"{jid}/{i}: {x}" for x in ae)
     if not ae:errors.extend(_finite_fields_errors([e]))
    except Exception as exc: errors.append(f"artifact validation failure {jid}/{i}: {exc}")
   try:
    d=load_segment(e); ti=d["times"]
    if any(not np.all(np.isfinite(d[k])) for k in ("times","V","T_avg","psi_avg","Ax")): errors.append(f"nonfinite field {jid}/{i}")
    if len(ti)!=8001 or not np.allclose(ti,np.arange(12000,20001),rtol=0,atol=1e-10): errors.append(f"retained schedule mismatch {jid}/{i}")
    r={"job_id":jid,"direction":"ascending" if jid.startswith("ascending") else "descending","seed":int(job["seed"]),"Je":float(e["Je"]),"V":meanwin(ti,d["V"],18000,20000),"T_avg":meanwin(ti,d["T_avg"],18000,20000),"psi_avg":meanwin(ti,d["psi_avg"],18000,20000),"V_integrated":integrated(d["Ax"],ti),"drift":meanwin(ti,d["V"],18000,20000)-meanwin(ti,d["V"],13000,15000)}
    if not all(math.isfinite(r[k]) for k in ("V","T_avg","psi_avg","V_integrated","drift")):errors.append(f"nonfinite observables {jid}/{i}")
    rows.append(r); bydir[r["direction"]][r["Je"]].append(r)
   except Exception as exc: errors.append(f"read failure {jid}/{i}: {exc}")
 # Per-direction stationarity and six ordered adjacent contrasts.
 stationarity={}; deltas={}; resolved=False
 for direction in DIRECTIONS:
  points=declared; stationarity[direction]={}
  for point in points:
   dr=[x["drift"] for x in bydir[direction][point]]; n=len(dr); sem=float(np.std(dr,ddof=1)/math.sqrt(n)) if n>1 else float("nan"); crit=float(t.ppf(.975,n-1)) if n>1 else float("nan"); half=crit*sem if math.isfinite(sem) else float("nan"); mean=float(np.mean(dr)) if dr else float("nan")
   stationarity[direction][str(point)]={"n":n,"mean":mean,"ci_low":mean-half,"ci_high":mean+half,"bound":.0025,"pass":n==4 and math.isfinite(half) and abs(mean)+half<=.0025}
  vals=[]
  for a,b in zip(points,points[1:]):
   av={x["seed"]:x["V"] for x in bydir[direction][a]}; bv={x["seed"]:x["V"] for x in bydir[direction][b]}
   if set(av)!=set(SEEDS) or set(bv)!=set(SEEDS): errors.append(f"incomplete paired current {direction}/{a}/{b}"); continue
   vals.append({"Je_low":a,"Je_high":b,"stats":stat([bv[s]-av[s] for s in SEEDS],True)})
  deltas[direction]=vals
  if all(x.get("pass") for x in stationarity[direction].values()) and any(vals[i]["stats"]["negative"] and vals[i+1]["stats"]["negative"] for i in range(len(vals)-1)): resolved=True
 complete=not errors and len(rows)==56
 verified=complete and validate
 if not verified:
  for direction in DIRECTIONS:
   for point in stationarity[direction].values():point['pass']=False
   for pair in deltas[direction]:pair['stats']['negative']=False
 sources=[Path(__file__),Path(__file__).with_name('compare_results.py'),Path(__file__).with_name('analyze_full_iv.py'),Path(__file__).with_name('analyze_full_curl_ndr.py')]
 result={"schema":"tdgl-full-curl-ndr-history-analysis-v1","complete":complete,"pending":not complete,"errors":errors,"artifact_validation":validate,"finite_saved_fields_validation":validate,"protocol":hp,"rows":rows,"stationarity":stationarity,"deltas":deltas,"stationary_branch_confirmed":False,"ordered_negative_pair_found":bool(verified and resolved),"provenance":{"protocol":str(protocol.resolve()),"protocol_sha256":sha(protocol),"campaign_root":str(campaign.resolve()),"campaign_sha256":sha(campaign/'campaign.json') if (campaign/'campaign.json').is_file() else None,"chain_receipt_sha256":{e['_chain_path']:sha(Path(e['_chain_path'])) for es in grouped.values() for e in es},"analysis_source_sha256":{p.name:sha(p) for p in sources},"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"h5py":h5py.__version__},"scope":"ordered-history descriptive analysis; frames are not independent replicates; stationary branch requires refinement and further convergence checks"}
 return finite_json(result)
def write(result,out):
 out.mkdir(parents=True,exist_ok=True); (out/"history_analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
 with (out/"history_seed_rows.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["job_id","direction","seed","Je","V","T_avg","psi_avg","V_integrated","drift"]); w.writeheader(); w.writerows(result["rows"])
 with (out/"history_deltas.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f); w.writerow(["direction","Je_low","Je_high","mean","upper95_bonferroni12","negative"])
  for d,vs in result["deltas"].items():
   for x in vs:w.writerow([d,x["Je_low"],x["Je_high"],x["stats"]["mean"],x["stats"]["upper95_bonferroni12"],x["stats"]["negative"]])
 try:
  import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
  fig,ax=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
  for direction,color in (("ascending","#0072B2"),("descending","#D55E00")):
   rr=[r for r in result["rows"] if r["direction"]==direction]; points=result["protocol"]["currents"]
   for key,axis,title in (("V",ax[0],"V"),("T_avg",ax[1],"T_avg"),("psi_avg",ax[2],"mean |ψ|²")):
    means=[]; lows=[]; highs=[]
    for je in points:
     vals=[r[key] for r in rr if r["Je"]==je]; means.append(np.mean(vals) if len(vals)==4 else np.nan); sem=np.std(vals,ddof=1)/math.sqrt(4) if len(vals)==4 else np.nan; half=float(t.ppf(.975,3))*sem; lows.append(means[-1]-half); highs.append(means[-1]+half)
    axis.plot(points,means,"o-",color=color,label=direction); axis.fill_between(points,lows,highs,color=color,alpha=.15)
    for seed in sorted({r["seed"] for r in rr}):
     one=[r for r in rr if r["seed"]==seed]; one.sort(key=lambda x:x["Je"]); axis.plot([r["Je"] for r in one],[r[key] for r in one],color=color,alpha=.22,lw=.7)
    axis.set(xlabel="Je",ylabel=title); axis.grid(alpha=.2)
  ax[0].legend(); fig.suptitle("Full-curl ordered histories · 20kτ dwell · final 18–20kτ\nFour seeds; mean ± nominal 95% CI; stationary negative branch unconfirmed"); fig.savefig(out/"history_V_T_psi.png",dpi=180); fig.savefig(out/"history_V_T_psi.pdf"); plt.close(fig)
 except ImportError: pass
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--protocol",required=True,type=Path);p.add_argument("--campaign-root",required=True,type=Path);p.add_argument("--out",required=True,type=Path);p.add_argument("--validate-artifacts",action="store_true");a=p.parse_args(argv);r=analyze(a.protocol,a.campaign_root,a.validate_artifacts);write(r,a.out);print(json.dumps({"complete":r["complete"],"out":str(a.out)}));return 0 if r["complete"] else 1
if __name__=="__main__":raise SystemExit(main())
