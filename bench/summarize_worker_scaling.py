#!/usr/bin/env python3
"""Gate all pilot pairs and summarize persistent-worker throughput."""
import argparse
import json
import pathlib
import statistics
from compare_results import load, meta_errors, validate_artifact, field_errors


def summarize(base, counts):
    groups=[]
    for count in counts:
        path=base/f"worker_pilot_{count}"/"worker_scaling.json"
        data=load(path)
        expected=len(data["selected_indices"])*data["samples_per_current"]*data["repetitions"]*2
        assert data["actual_nworkers"]==count and len(data["rows"])==expected
        assert all(r["complete"] and r["status"] in ("Success","success","ReturnCode.Success") for r in data["rows"])
        pairs={}
        for row in data["rows"]:
            key=(row["repetition"],row["current_index"],row["sample"])
            assert row["backend"] not in pairs.setdefault(key,{})
            pairs[key][row["backend"]]=row
        maximum_error=0.;artifacts=[]
        for key,pair in pairs.items():
            rp=pathlib.Path(pair["reference"]["output_path"]).with_suffix(".json")
            cp=pathlib.Path(pair["cpu_compatible"]["output_path"]).with_suffix(".json")
            r,c=load(rp),load(cp)
            errors=meta_errors(r,c)
            er,rh=validate_artifact(rp,r);ec,ch=validate_artifact(cp,c)
            errors+=er+ec
            assert not errors,(count,key,errors)
            errors,ferr=field_errors(rh,ch)
            assert not errors and set(ferr)=={"times","V","psi_avg","T_avg","psi_real","psi_imag","T","Ax","Ay"}
            assert all(v<=1e-11 for v in ferr.values()),(count,key,ferr)
            assert all(r[k]==c[k] for k in ("accepted_steps","rejected_steps","rhs_calls","initial_vector_hash"))
            maximum_error=max(maximum_error,max(ferr.values()))
            artifacts.append(dict(repetition=key[0],index=key[1],sample=key[2],seed=r["seed"],
                initial_hash=r["initial_hash"],reference_output_hash=r["output_hash"],candidate_output_hash=c["output_hash"],field_errors=ferr))
        phases={b:[p["total_phase_wall_seconds"] for p in data["phases"] if p["backend"]==b] for b in ("reference","cpu_compatible")}
        ratios=[a/b for a,b in zip(phases["reference"],phases["cpu_compatible"])]
        groups.append(dict(workers=count,paired_trajectories=len(pairs),horizon=data["horizon"],noise=data["noise"],
            jobs_per_phase=expected//(data["repetitions"]*2),phase_samples_seconds=phases,
            phase_medians_seconds={b:statistics.median(v) for b,v in phases.items()},
            paired_speedup_median=statistics.median(ratios),paired_speedup_min=min(ratios),paired_speedup_max=max(ratios),
            max_field_error=maximum_error,source_file=str(path),all_seeds=data["all_seeds"],artifacts=artifacts))
    baseline=next(g for g in groups if g["workers"]==8)["phase_medians_seconds"]["reference"]
    return dict(scope="100-time-unit pilot; 36 independent full-output jobs per phase; not a 5000-unit full scan",
        valid=True,groups=groups,normal_8_reference_seconds=baseline,
        optimized_24_vs_normal_8=baseline/next(g for g in groups if g["workers"]==24)["phase_medians_seconds"]["cpu_compatible"],
        max_field_error=max(g["max_field_error"] for g in groups),
        total_pairs=sum(g["paired_trajectories"] for g in groups))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("runroot",type=pathlib.Path);p.add_argument("--out",type=pathlib.Path,required=True)
    args=p.parse_args()
    result=summarize(args.runroot,[1,6,8,12,24])
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("valid","total_pairs","max_field_error","optimized_24_vs_normal_8")}))
