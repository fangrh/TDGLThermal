#!/usr/bin/env python3
"""Four-current, finite-width NDR interval analysis."""
from __future__ import annotations
import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_full_curl_ndr import (read_chains, read_chains_with_errors, read_segment,
    _complete_chain, _mean_window, validate_chain_artifacts, _sha256)

SEEDS = tuple(range(2026110411, 2026110419)); A = .3 * 68 / 98
CURRENTS = tuple(A + i * (.3 / 98) / 3 for i in range(4))
T_CRIT_BONFERRONI = 2.641859670498041


def _stats(values: list[float], t=T_CRIT_BONFERRONI) -> dict[str, Any]:
    n = len(values); mean = float(np.mean(values)) if n else float("nan")
    sem = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    upper = mean + t * sem if math.isfinite(sem) else float("nan")
    return {"n": n, "values": values, "mean": mean, "sem": sem, "upper95": upper,
            "tcrit": t, "pass": n == 8 and math.isfinite(upper)}


def _stationarity(values: list[float]) -> dict[str, Any]:
    n = len(values); mean = float(np.mean(values)) if n else float("nan")
    sem = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    half = 2.364624 * sem if math.isfinite(sem) else float("nan")
    return {"n": n, "mean": mean, "sem": sem, "ci_low": mean-half, "ci_high": mean+half,
            "bound": .0025, "pass": n == 8 and math.isfinite(half) and mean-half >= -.0025 and mean+half <= .0025}


def adjacent_gate(points: list[dict[str, Any]], stationarity_ok: bool = True) -> dict[str, Any]:
    """Require two consecutive negative adjacent contrasts and total upper < -.01."""
    if len(points) != 4 or any(len(p.get("values", [])) != 8 for p in points):
        return {"resolved": False, "classification": "incomplete seed matrix", "edges": []}
    edges = []
    for i in range(3):
        vals = [b - a for a, b in zip(points[i]["values"], points[i + 1]["values"])]
        edges.append(_stats(vals))
    candidates = []
    for i in range(2):
        pair = edges[i:i + 2]
        total = [points[i + 2]["values"][j] - points[i]["values"][j] for j in range(8)]
        total_stat = _stats(total)
        sum_upper = sum(e["upper95"] for e in pair)
        candidates.append({"edges": [i, i + 1], "total_descriptive": total_stat,
                           "simultaneous_total_upper": sum_upper,
                           "pass": all(e["upper95"] < 0 for e in pair) and sum_upper < -.01})
    resolved = stationarity_ok and any(c["pass"] for c in candidates)
    return {"edges": edges, "two_edge_candidates": candidates, "resolved": resolved,
            "stationarity_ok": stationarity_ok,
            "classification": "negative-slope interval supported in the preparation ensemble" if resolved else ("unresolved transition" if any(e["mean"] < 0 for e in edges) else "unresolved")}


def _entry_key(e: dict[str, Any]) -> tuple[float, int]:
    return (float(e.get("Je", math.nan)), int(e.get("seed", -1)))


def _logical(entries: list[dict[str, Any]], start: int = 1) -> list[dict[str, Any]]:
    out = []
    for e in entries:
        x = dict(e); x["original_segment"] = e.get("segment", e.get("step"))
        x["segment"] = int(round(float(e.get("cumulative_end", 0)) / 5000.0))
        out.append(x)
    return sorted(out, key=lambda x: x["segment"])


def _protocol_ok(e: dict[str, Any], je: float) -> bool:
    m = e.get("manifest", {})
    return math.isclose(float(e.get("Je", math.nan)), je, abs_tol=1e-15) and m.get("complete") is True and m.get("status") in ("success", "Success", "ReturnCode.Success") and all(math.isclose(float(m.get(k, math.nan)), v, abs_tol=1e-12) for k, v in (("horizon", 5000.), ("skip_ratio", .6), ("noise", .001), ("tolerance", 1e-4)))


def analyze(endpoint_root: Path, screen_root: Path | None, interior_root: Path | None, history_root: Path | None = None, validate_artifacts: bool = False) -> dict[str, Any]:
    ep, errors = read_chains_with_errors(endpoint_root) if "read_chains_with_errors" in globals() else (read_chains(endpoint_root), [])
    interior, ie = read_chains_with_errors(interior_root) if interior_root else ({}, [])
    errors += ie
    screen, se = read_chains_with_errors(screen_root) if screen_root else ({}, [])
    errors += se
    by: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for source in (ep, screen, interior):
        for rows in source.values():
            if not rows: continue
            je, seed = _entry_key(rows[0])
            canonical = min(CURRENTS, key=lambda j: abs(j-je))
            if abs(canonical-je) > 1e-12: continue
            by[(canonical, seed)] += _logical(rows)
    points = []; complete = True; artifacts = []
    for je in CURRENTS:
        vals=[]; row_meta=[]
        for seed in SEEDS:
            rows = by.get((je, seed), []); rows.sort(key=lambda x: x["segment"])
            good = (_complete_chain(rows, expected_je=je) and
                    all(_protocol_ok(x, je) and int(x.get("seed", -1)) == seed and
                        int(x.get("manifest", {}).get("seed", -1)) == seed for x in rows))
            if not good: complete = False; continue
            if validate_artifacts: artifacts += validate_chain_artifacts(rows)
            try:
                segs = [read_segment(x) for x in rows]
                final = segs[-1]; v = _mean_window(final["times"] + 45000, final["V"], 48000, 50000)
                row_meta.append({"seed": seed, "V": v, "T_avg": _mean_window(final["times"] + 45000, final["T_avg"], 48000, 50000), "psi_avg": _mean_window(final["times"] + 45000, final["psi_avg"], 48000, 50000), "V_integrated": final["V_integrated"]})
                vals.append(v)
            except (OSError, KeyError, ValueError, IndexError) as exc: errors.append(f"{je}/{seed}: {exc}"); complete = False
        points.append({"Je": je, "values": vals, "rows": row_meta, "stats": _stats(vals)})
    stationarity = {}
    stationarity_ok = complete
    for point in points:
        drifts=[]
        for seed in SEEDS:
            rows=sorted(by.get((point["Je"], seed), []), key=lambda x:x["segment"])
            if len(rows)==10:
                try:
                    s=[read_segment(x) for x in rows]; late=_mean_window(s[-1]["times"]+45000,s[-1]["V"],48000,50000); prev=_mean_window(s[-2]["times"]+40000,s[-2]["V"],43000,45000); drifts.append(late-prev)
                except (OSError, KeyError, ValueError, IndexError): pass
        stationarity[str(point["Je"])] = _stationarity(drifts)
        stationarity_ok &= stationarity[str(point["Je"])] ["pass"]
    complete = complete and not errors and not artifacts
    gate = adjacent_gate(points, stationarity_ok and complete) if complete else {"resolved": False, "classification": "unresolved", "edges": []}
    history = analyze_history(history_root, validate_artifacts) if history_root else None
    return {"schema":"tdgl-ndr-interval-analysis-v1", "model_units":"native model units", "currents":list(CURRENTS), "complete":complete and not errors and not artifacts, "errors":errors + artifacts, "artifact_validation":validate_artifacts, "points":points, "stationarity":stationarity, "gate":gate, "history":history, "screen":analyze_screen(screen_root) if screen_root else None}


def analyze_screen(root: Path) -> dict[str, Any]:
    groups=read_chains(root); rows=[]
    for job, entries in groups.items():
        if not entries: continue
        try:
            s=read_segment(entries[0]); rows.append({"id":job,"Je":entries[0].get("Je"),"seed":entries[0].get("seed"),"V":_mean_window(s["times"],s["V"],2000,5000),"T_avg":_mean_window(s["times"],s["T_avg"],2000,5000),"psi_avg":_mean_window(s["times"],s["psi_avg"],2000,5000)})
        except (OSError, KeyError, ValueError): pass
    return {"rows":rows,"horizon":5000.0}


def analyze_history(root: Path, validate_artifacts: bool = False) -> dict[str, Any]:
    groups=defaultdict(list); errors=[]
    for path in root.glob("**/step*/chain.json"):
        try:
            e=json.loads(path.read_text(encoding="utf-8")); e["_chain_path"]=str(path); e["_segment_dir"]=str(path.parent); groups[str(e.get("id",path.parent.parent.name))].append(e)
        except (OSError, ValueError, TypeError) as exc: errors.append(f"{path}: {exc}")
    rows=[]
    for job, entries in groups.items():
        try:
            direction=job.split("_seed")[0]; seed=int(entries[0].get("seed"))
            if direction not in ("ascending", "descending") or seed not in SEEDS[:3]:
                errors.append(f"unexpected history job {job}"); continue
            entries=sorted(entries,key=lambda x:int(x.get("step",0)))
            if [e.get("step") for e in entries] != [1,2,3,4]:
                errors.append(f"incomplete history {job}"); continue
            expected=CURRENTS if direction=="ascending" else CURRENTS[::-1]
            adapted=[]
            for step,(e,je) in enumerate(zip(entries,expected),1):
                if not _protocol_ok(e,je) or e.get("seed")!=seed or not math.isclose(e.get("cumulative_end",0),50000+step*5000,abs_tol=1e-8):
                    errors.append(f"history protocol mismatch {job}/{step}")
                a=dict(e);a["segment"]=step;a["preceding_rng_path"]=e.get("preceding_rng")
                a["preceding_sha256"]=e["manifest"].get("initial_source_hash");adapted.append(a)
                s=read_segment(e);v=s["V"];half=len(v)//2
                rows.append({"id":job,"direction":direction,"seed":seed,"step":step,"Je":e.get("Je"),
                             "V":float(np.mean(v)),"T_avg":float(np.mean(s["T_avg"])),"psi_avg":float(np.mean(s["psi_avg"])),
                             "V_integrated":s["V_integrated"],"late_half_drift":float(np.mean(v[half:])-np.mean(v[:half]))})
            if validate_artifacts:
                errors.extend(validate_chain_artifacts(adapted))
                first=entries[0];m=first["manifest"]
                if _sha256(Path(first["preceding_hdf5"]))!=m.get("initial_source_hash"):
                    errors.append(f"history initial field hash mismatch {job}")
                if _sha256(Path(first["preceding_rng"]))!=first.get("rng_before_sha256"):
                    errors.append(f"history initial RNG hash mismatch {job}")
        except (OSError, KeyError, ValueError, TypeError) as exc: errors.append(f"history {job}: {exc}")
    expected_jobs={f"{d}_seed{s}" for d in ("ascending","descending") for s in SEEDS[:3]}
    complete=set(groups)==expected_jobs and len(rows)==24 and not errors
    return {"rows":rows,"complete":complete,"errors":errors,"artifact_validation":validate_artifacts,
            "scope":"descriptive ordered-current history; three seeds per direction"}


def _finite(x):
    if isinstance(x,float) and not math.isfinite(x): return None
    if isinstance(x,dict): return {k:_finite(v) for k,v in x.items()}
    if isinstance(x,list): return [_finite(v) for v in x]
    return x


def write_outputs(result: dict[str,Any], out: Path):
    out.mkdir(parents=True,exist_ok=True); result=_finite(result); (out/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (out/"interval_points.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["Je","seed","V","T_avg","psi_avg","V_integrated"])
        for p in result["points"]:
            for r in p.get("rows",[]): w.writerow([p["Je"],r["seed"],r["V"],r["T_avg"],r["psi_avg"],r["V_integrated"]])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True); pts=result["points"]
    if all(len(p.get("values",[]))==8 for p in pts):
        current=np.array([p["Je"] for p in pts]);values=np.array([p["values"] for p in pts]);means=values.mean(axis=1)
        for seed in range(8):ax[0].plot(current,values[:,seed],color="gray",alpha=.35,lw=.8)
        ax[0].errorbar(current,means,yerr=[means-values.min(axis=1),values.max(axis=1)-means],fmt="o-",capsize=4,label="Mean and seed range")
        ax[0].legend(fontsize=8)
    else:
        ax[0].text(.5,.5,"Four-point campaign incomplete",ha="center",transform=ax[0].transAxes)
    ax[0].set_title("Independent preparations: 50000τ"); ax[0].set_xlabel("Je (model units)"); ax[0].set_ylabel("V (model units)")
    gate=result["gate"]; es=gate.get("edges",[])
    if es:
        mean=[e["mean"] for e in es];upper=[e["upper95"] for e in es]
        ax[1].vlines(range(3),mean,upper,color="gray")
        ax[1].plot(range(3),mean,"o",label="Mean paired ΔV")
        ax[1].plot(range(3),upper,"v",color="#D55E00",label="Simultaneous 95% upper")
        ax[1].set_xticks(range(3),["A → 1","1 → 2","2 → B"]);ax[1].legend(fontsize=8)
    ax[1].axhline(0,color="k",ls=":"); ax[1].set_title("One-sided bounds: three contrasts"); ax[1].set_xlabel("Adjacent current interval"); ax[1].set_ylabel("ΔV (model units)")
    hist=(result.get("history") or {}).get("rows",[])
    for direction,style,color in (("ascending","o-","#0072B2"),("descending","s--","#D55E00")):
        for index,seed in enumerate(sorted({r["seed"] for r in hist if r["direction"]==direction})):
            rr=sorted([r for r in hist if r["direction"]==direction and r["seed"]==seed],key=lambda r:r["step"])
            ax[2].plot([r["Je"] for r in rr],[r["V"] for r in rr],style,color=color,alpha=.7,label=direction.capitalize() if index==0 else None)
    ax[2].set_title("Ordered histories: 5000τ dwell"); ax[2].set_xlabel("Je (model units)"); ax[2].set_ylabel("V (model units)")
    if hist:ax[2].legend(fontsize=8)
    for axis in ax:axis.grid(alpha=.2)
    fig.suptitle("Corrected full curl · T0 = 0.18 · legacy noise = 0.001",fontsize=13)
    fig.savefig(out/"ndr_interval.png",dpi=180); fig.savefig(out/"ndr_interval.pdf"); plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--endpoint-root",required=True,type=Path); p.add_argument("--screen-root",type=Path); p.add_argument("--interior-root",required=True,type=Path); p.add_argument("--history-root",type=Path); p.add_argument("--out",required=True,type=Path); p.add_argument("--validate-artifacts",action="store_true"); a=p.parse_args(argv); r=analyze(a.endpoint_root,a.screen_root,a.interior_root,a.history_root,a.validate_artifacts); write_outputs(r,a.out); print(json.dumps({"complete":r["complete"],"resolved":r["gate"]["resolved"],"out":str(a.out)})); return 0
if __name__ == "__main__": raise SystemExit(main())
