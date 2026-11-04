#!/usr/bin/env python3
"""Bounded, resumable CPU benchmark campaign scheduler.

The scheduler is intentionally stdlib-only.  Numerical comparison is delegated
to ``compare_results.py`` in the configured Python environment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def je_dir(value: float) -> str:
    return f"Je{value:g}"


def selected_backends(mode: str) -> str:
    return {"paired": "reference,cpu_compatible", "reference": "reference",
            "cpu_compatible": "cpu_compatible"}[mode]


def benchmark_command(args: argparse.Namespace, current: float, seed: int,
                      stage_out: pathlib.Path, initial: pathlib.Path | None) -> list[str]:
    cmd = [args.julia, "--project=" + args.project, str(args.benchmark),
           "--out", str(stage_out), "--horizon", str(args.horizon),
           "--currents", str(current), "--seed", str(seed), "--noise", str(args.noise),
           "--repetitions", "1", "--warmup-horizon", "2", "--backends",
           selected_backends(args.backend), "--skip", str(args.skip),
           "--tolerance", str(args.tolerance)]
    if initial is None:
        cmd += ["--preparation", "configured_down"]
    else:
        cmd += ["--initial-state", str(initial)]
    return cmd


def compare_command(args: argparse.Namespace, stage_out: pathlib.Path, current: float, seed: int) -> list[str]:
    root = stage_out / je_dir(current) / "rep1"
    return [sys.executable, str(args.comparator), "--reference",
            str(root / f"reference_Je{current}_seed{seed}.json"), "--candidate",
            str(root / f"cpu_compatible_Je{current}_seed{seed}.json"), "--out",
            str(stage_out / "comparison.csv")]


def run_process(cmd: list[str], out_log: pathlib.Path, err_log: pathlib.Path,
                timeout: float) -> tuple[int, bool]:
    out_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_log.open("w", encoding="utf-8", errors="replace") as out, err_log.open(
                "w", encoding="utf-8", errors="replace") as err:
            p = subprocess.run(cmd, stdout=out, stderr=err, shell=False,
                               timeout=max(0.01, timeout), check=False)
        return int(p.returncode), False
    except subprocess.TimeoutExpired:
        return -1, True


def run_job(args: argparse.Namespace, current: float, seed: int,
            deadline: float) -> dict[str, Any]:
    job_start = time.monotonic()
    job_root = pathlib.Path(args.out) / f"Je{current:g}" / f"seed{seed}"
    segments: list[dict[str, Any]] = []
    previous: pathlib.Path | None = pathlib.Path(args.initial_state) if args.initial_state else None
    status = "complete"
    for segment in range(1, args.segments + 1):
        if time.monotonic() >= deadline:
            status = "deferred" if not segments else "incomplete"
            break
        stage = job_root / f"segment{segment}"
        seg_seed = args.seed_start + (seed - args.seed_start) + segment - 1
        cmd = benchmark_command(args, current, seg_seed, stage, previous)
        remaining = deadline - time.monotonic()
        started = time.monotonic()
        rc, timed_out = run_process(cmd, stage / "benchmark.stdout.log",
                                     stage / "benchmark.stderr.log", remaining)
        rec: dict[str, Any] = {"segment": segment, "seed": seg_seed,
                               "command": cmd, "returncode": rc,
                               "timed_out": timed_out,
                               "elapsed_seconds": time.monotonic() - started}
        segments.append(rec)
        if rc != 0:
            status = "incomplete" if timed_out else "failed"
            break
        root = stage / je_dir(current) / "rep1"
        if args.backend == "paired":
            cmp_cmd = compare_command(args, stage, current, seg_seed)
            cmp_started = time.monotonic()
            cmp_rc, cmp_timeout = run_process(cmp_cmd, stage / "compare.stdout.log",
                                              stage / "compare.stderr.log",
                                              max(0.01, deadline - time.monotonic()))
            rec.update(compare_command=cmp_cmd, compare_returncode=cmp_rc,
                       compare_timed_out=cmp_timeout,
                       compare_elapsed_seconds=time.monotonic() - cmp_started)
            if cmp_rc != 0:
                status = "failed" if not cmp_timeout else "incomplete"
                break
            previous = root / f"reference_Je{current}_seed{seg_seed}.h5"
        else:
            previous = root / f"{args.backend}_Je{current}_seed{seg_seed}.h5"
    return {"current": current, "seed": seed, "status": status,
            "segments": segments, "elapsed_seconds": time.monotonic() - job_start}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--julia", required=True); p.add_argument("--project", default=".")
    p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=1)
    p.add_argument("--horizon", type=float, default=5000); p.add_argument("--currents", default="0.06,0.18,0.27")
    p.add_argument("--seed-start", type=int, default=2026110401); p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--segments", type=int, default=1); p.add_argument("--backend", choices=("paired", "reference", "cpu_compatible"), default="paired")
    p.add_argument("--skip", type=float, default=.9); p.add_argument("--campaign-hours", type=float, default=24)
    p.add_argument("--tolerance", type=float, default=1e-4); p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--initial-state"); p.add_argument("--benchmark", default=str(pathlib.Path(__file__).with_name("benchmark_cpu.jl")))
    p.add_argument("--comparator", default=str(pathlib.Path(__file__).with_name("compare_results.py")))
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not pathlib.Path(args.julia).is_absolute():
        parser().error("--julia must be an absolute path")
    if args.workers < 1 or args.seeds < 1 or args.segments < 1 or args.campaign_hours <= 0:
        parser().error("workers, seeds, segments and campaign-hours must be positive")
    currents = [float(x) for x in args.currents.split(",") if x.strip()]
    if not currents:
        parser().error("--currents must contain at least one value")
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    started_at, started = iso_now(), time.monotonic()
    deadline = started + args.campaign_hours * 3600
    jobs = [(c, args.seed_start + i) for c in currents for i in range(args.seeds)]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for c, s in jobs:
            if time.monotonic() >= deadline:
                results.append({"current": c, "seed": s, "status": "deferred", "segments": [], "elapsed_seconds": 0.0})
                continue
            futures[pool.submit(run_job, args, c, s, deadline)] = (c, s)
        for f in concurrent.futures.as_completed(futures):
            result = f.result(); results.append(result)
            print(f"Je{result['current']:g} seed{result['seed']}: {result['status']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    counts = {k: sum(r["status"] == k for r in results) for k in ("complete", "failed", "incomplete", "deferred")}
    summary = {"started_at": started_at, "ended_at": iso_now(), "elapsed_seconds": time.monotonic() - started,
               "arguments": {k: v for k, v in vars(args).items() if not k.startswith("_")},
               "performance_phase": ("paired_validation" if args.backend == "paired" else
                                      f"controlled_{args.backend}_only_no_accuracy_claim"),
               "environment": {"cwd": os.getcwd(), "python": sys.executable}, "counts": counts,
               "jobs": sorted(results, key=lambda r: (r["current"], r["seed"]))}
    pathlib.Path(args.out, "campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if counts["failed"] == counts["incomplete"] == 0 and counts["deferred"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
