"""Prepare a full-current-range campaign, retaining compatible prior checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-run-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--config", default="studies/full_curl_ndr/config_T0.18.yaml")
    p.add_argument("--seeds", default="2026110411")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    model = Path(__file__).resolve().parents[1] / "current_sweep_thermal_nofastscan.jl"
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    imported = {}
    for name in ("ndr_endpoint_50000_v1", "ndr_screen_5000_v1", "ndr_interior_50000_v1"):
        for path in (args.source_run_root / name).glob("*/segment*/chain.json"):
            e = load(path); m = e["manifest"]
            if e["seed"] not in seeds: continue
            if m["config_hash"] != cfg_hash or m["model_hash"] != model_hash: continue
            if (m["horizon"], m["skip_ratio"], m["noise"], m["tolerance"]) != (5000, .6, .001, 1e-4): continue
            if not m["complete"] or not Path(m["output_path"]).is_file() or not Path(e["rng_after_path"]).is_file(): continue
            imported.setdefault((float(e["Je"]), int(e["seed"])), []).append((e, path.resolve()))
    a, b = .3 * 68 / 98, .3 * 69 / 98
    base = [.3 * i / 98 for i in range(99)]
    refinements = [a + (b-a)/3, a + 2*(b-a)/3]
    points, jobs = [], []
    for kind, currents in (("base", base), ("refinement", refinements)):
        for index, current in enumerate(currents):
            for seed in seeds:
                ident = f"{kind}_i{index:03d}_seed{seed}"
                matches = [v for (j,s),v in imported.items() if s == seed and abs(j-current) < 1e-12]
                entries = sorted(matches[0], key=lambda x: x[0]["cumulative_end"]) if matches else []
                ends = [e["cumulative_end"] for e,_ in entries]
                if ends != list(range(5000, 5000*(len(ends)+1), 5000)):
                    raise ValueError(f"incomplete or conflicting imported chain: {ident}")
                prior = int(ends[-1]) if ends else 0
                remaining = (50000-prior)//5000
                if remaining < 0: raise ValueError("imported horizon exceeds target")
                job = {"id":ident,"config_path":args.config,"Je":current,"seed":seed,
                       "segments":remaining,"prior_cumulative_horizon":prior}
                if entries:
                    last = entries[-1][0]
                    job.update(initial_state_path=last["manifest"]["output_path"], initial_rng_path=last["rng_after_path"])
                jobs.append(job)
                points.append({"id":ident,"Je":current,"seed":seed,"kind":kind,
                               "existing_chain_paths":[str(path) for _,path in entries],"segments_to_run":remaining})
    args.out.mkdir(parents=True, exist_ok=True)
    job_path = (args.out / "jobs.json").resolve()
    catalog = {"schema":"tdgl-full-iv-catalog-v1", "protocol":{"temperature":.18,"noise":.001,
               "horizon":50000,"segment_horizon":5000,"snapshot_interval":1,"skip_ratio":.6,"tolerance":1e-4,
               "seeds":seeds,"base_currents":base,"refinement_currents":refinements},"points":points,
               "jobs_path":str(job_path),"model_hash":model_hash,"config_hash":cfg_hash,
               "scope":"Independent hot preparations; global seed count is explicit; local eight-seed overlay is separate."}
    for path, value in ((job_path,jobs),(args.out/"catalog.json",catalog)):
        if path.exists(): raise FileExistsError(path)
        path.write_text(json.dumps(value,indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"points":len(points),"segments_to_run":sum(j["segments"] for j in jobs),
                      "fully_reused":sum(j["segments"]==0 for j in jobs),"partially_reused":sum(0<j["segments"]<10 for j in jobs)}))


if __name__ == "__main__":
    main()
