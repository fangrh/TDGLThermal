#!/usr/bin/env python3
"""Summarize actual paired CPU artifacts; comparison validity is mandatory."""
import argparse,json,pathlib,subprocess,sys,statistics,csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    root,out=pathlib.Path(args.root),pathlib.Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    comparator=pathlib.Path(__file__).with_name("compare_results.py")
    rows=[]
    for tag in ("cpu_v2_deterministic","cpu_v2_noisy"):
        for je_dir in sorted((root/tag).glob("Je*")):
            for rep in sorted(je_dir.glob("rep*")):
                ref=next(rep.glob("reference*.json"))
                cand=next(rep.glob("cpu_compatible*.json"))
                target=out/tag/je_dir.name/rep.name
                target.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,str(comparator),"--reference",str(ref),
                     "--candidate",str(cand),"--out",str(target/"comparison.csv")]
                if rep.name=="rep1":
                    cmd+=["--plot",str(target/"traces.png")]
                run=subprocess.run(cmd,capture_output=True,text=True)
                if run.returncode:
                    raise RuntimeError(f"Comparison failed: {rep}\n{run.stdout}\n{run.stderr}")
                result=json.loads(run.stdout.strip().splitlines()[-1])
                if not result["passed"] or not result.get("field_errors"):
                    raise RuntimeError(f"Unverified comparison: {rep}")
                a,b=json.loads(ref.read_text()),json.loads(cand.read_text())
                rows.append(dict(cohort=tag,Je=a["Je"],repetition=rep.name,
                    reference_seconds=a["wall_seconds"],candidate_seconds=b["wall_seconds"],
                    speedup=a["wall_seconds"]/b["wall_seconds"],
                    max_field_error=max(result["field_errors"].values()),
                    reference_rhs_bytes=a["rhs_allocated"],candidate_rhs_bytes=b["rhs_allocated"],
                    rhs_calls_equal=a["rhs_calls"]==b["rhs_calls"],
                    horizon=a["horizon"],noise=a["noise"],source_hash=a["model_hash"],
                    candidate_source_hash=a["rhs_source_hash"]))
    if len(rows)!=30:
        raise RuntimeError(f"Expected30 paired measurements, found{len(rows)}")
    groups=[]
    for tag in ("cpu_v2_deterministic","cpu_v2_noisy"):
        for je in sorted({r["Je"] for r in rows if r["cohort"]==tag}):
            rr=[r for r in rows if r["cohort"]==tag and r["Je"]==je]
            values=[r["speedup"] for r in rr]
            groups.append(dict(cohort=tag,Je=je,n=len(rr),
                median_speedup=statistics.median(values),min_speedup=min(values),
                max_speedup=max(values),max_field_error=max(r["max_field_error"] for r in rr)))
    result=dict(scope="100tau paired full-output CPU measurements; not complete production promotion",
        pair_count=len(rows),all_fields_exact=all(r["max_field_error"]==0 for r in rows),
        max_field_error=max(r["max_field_error"] for r in rows),
        all_rhs_counts_equal=all(r["rhs_calls_equal"] for r in rows),
        groups=groups,measurements=rows)
    (out/"cpu_v2_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (out/"cpu_v2_measurements.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
    for ax,tag,title in zip(axes,("cpu_v2_deterministic","cpu_v2_noisy"),
            ("Noise off; nontrivial cool preparation","Configured noise; normal preparation")):
        gg=[g for g in groups if g["cohort"]==tag]
        xx=[g["Je"] for g in gg]; yy=[g["median_speedup"] for g in gg]
        ax.errorbar(xx,yy,yerr=[[g["median_speedup"]-g["min_speedup"] for g in gg],
            [g["max_speedup"]-g["median_speedup"] for g in gg]],fmt="o-",capsize=4)
        ax.axhline(1,color="0.5",ls="--")
        ax.set_title(title,fontsize=10)
        ax.set_xlabel("Current density (model units)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Reference / optimized wall time")
    fig.suptitle("Matched 100τ full-output measurements: 5 paired runs per current",fontsize=11)
    fig.tight_layout()
    fig.savefig(out/"cpu_v2_speedup.png",dpi=220)
    fig.savefig(out/"cpu_v2_speedup.pdf")
    plt.close(fig)
    print(json.dumps({k:result[k] for k in ("pair_count","all_fields_exact","max_field_error","all_rhs_counts_equal","groups")}))
if __name__=="__main__":
    main()

