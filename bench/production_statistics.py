#!/usr/bin/env python3
"""Paired trajectory statistics for the validated production campaign.

Only the small scalar series V, T_avg, and psi_avg are read from each HDF5
output.  Full fields are intentionally never loaded.  This is a descriptive
uncertainty report; it is not a replacement for a reference-tolerance gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any

import h5py
import numpy as np

SIGNALS = ("V", "T_avg", "psi_avg")
TCRIT_95_DF7 = 2.364624


def _json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tau_initial_positive(x: np.ndarray) -> tuple[float, int]:
    y = np.asarray(x, dtype=float) - float(np.mean(x))
    if len(y) < 3 or float(np.dot(y, y)) == 0.0:
        return 0.5, 1
    ac = np.correlate(y, y, mode="full")[len(y) - 1:] / float(np.dot(y, y))
    positive = 0
    for value in ac[1:]:
        if value <= 0.0:
            break
        positive += 1
    tau = 0.5 + float(np.sum(ac[1:positive + 1]))
    return max(0.5, tau), positive


def _series(path: pathlib.Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        out = {name: np.asarray(h5[name][:], dtype=float) for name in SIGNALS}
    if any(a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all() for a in out.values()):
        raise ValueError(f"nonfinite or invalid scalar series in {path}")
    return out


def analyze(receipt_path: pathlib.Path) -> dict[str, Any]:
    receipt = _json(receipt_path)
    if receipt.get("valid") is not True or len(receipt.get("pairs", [])) != 24:
        raise ValueError("input artifact receipt is not a valid 24-pair receipt")
    receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    grouped: dict[float, list[dict[str, Any]]] = {}
    for pair in receipt["pairs"]:
        grouped.setdefault(float(pair["current"]), []).append(pair)
    if sorted(grouped) != [0.06, 0.18, 0.27] or any(len(v) != 8 for v in grouped.values()):
        raise ValueError("receipt does not contain exactly 3 currents with 8 seeds each")

    result: dict[str, Any] = {"schema": "tdgl-production-statistics-v1", "valid": True,
                              "input_receipt": {"path": str(receipt_path), "sha256": receipt_hash},
                              "signals": list(SIGNALS), "student_t": {"n": 8, "df": 7, "tcrit_95": TCRIT_95_DF7},
                              "rms_margin_rule": "descriptive 1% RMS lower-bound only; no scientific equivalence claim",
                              "reference_tolerance_gate": {"status": "pending", "reason": "reference-only tolerance uncertainty was not supplied"},
                              "currents": []}
    for current in sorted(grouped):
        pairs = sorted(grouped[current], key=lambda p: p["seed"])
        row: dict[str, Any] = {"Je": current, "n_trajectories": 8, "signals": {}, "pairs": []}
        loaded: list[tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]] = []
        for pair in pairs:
            by_label = {}
            for manifest in pair["manifests"]:
                by_label[manifest["label"]] = _series(pathlib.Path(manifest["output"]["output_path"]))
            loaded.append((pair, by_label["reference"], by_label["candidate"]))
            row["pairs"].append({"seed": pair["seed"], "reference_path": by_label["reference"].get("_path", pair["manifests"][0]["output"]["output_path"]), "candidate_path": pair["manifests"][1]["output"]["output_path"], "comparison_field_errors": pair["comparison"]["field_errors"]})
        for signal in SIGNALS:
            refs = [x[1][signal] for x in loaded]; cands = [x[2][signal] for x in loaded]
            if any(a.shape != b.shape for a, b in zip(refs, cands)):
                raise ValueError(f"shape mismatch for {current} {signal}")
            direct_equal = all(np.array_equal(a, b) for a, b in zip(refs, cands))
            pooled = np.concatenate(refs)
            pooled_mean, pooled_rms = float(np.mean(pooled)), float(np.std(pooled, ddof=0))
            means_ref = [float(np.mean(a)) for a in refs]
            means_cand = [float(np.mean(a)) for a in cands]
            std_ref = [float(np.std(a, ddof=1)) for a in refs]
            std_cand = [float(np.std(a, ddof=1)) for a in cands]
            diffs = np.asarray(means_cand) - np.asarray(means_ref)
            mean_diff = float(np.mean(diffs)); sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            half = TCRIT_95_DF7 * sd_diff / math.sqrt(8.0)
            ac_rows = []
            for a in refs:
                tau, positive_lags = _tau_initial_positive(a)
                block = max(1, int(math.ceil(10.0 * tau)))
                ac_rows.append({"n_samples": int(len(a)), "tau_int_initial_positive_pairs": tau, "positive_lags": positive_lags, "block_length": block, "block_count": int(len(a) // block)})
            row["signals"][signal] = {"meaning": "psi_avg is mean |psi|^2 over the saved trajectory", "pooled_reference_mean": pooled_mean, "pooled_reference_rms": pooled_rms, "one_percent_rms_lower_bound": 0.01 * pooled_rms, "reference_trajectory_mean": means_ref, "candidate_trajectory_mean": means_cand, "reference_trajectory_std": std_ref, "candidate_trajectory_std": std_cand, "paired_mean_difference_candidate_minus_reference": mean_diff, "paired_95pct_CI": [mean_diff - half, mean_diff + half], "direct_series_array_equal": direct_equal, "all_actual_paired_series_finite": True, "autocorrelation": {"method": "initial positive pairs", "reference_trajectories": ac_rows, "all_have_at_least_10_blocks": all(x["block_count"] >= 10 for x in ac_rows), "adequate_for_block_summary": all(x["block_count"] >= 10 for x in ac_rows)}}
        row["pairs"] = [{"seed": pair["seed"], "reference": pair["manifests"][0]["output"], "candidate": pair["manifests"][1]["output"]} for pair in pairs]
        result["currents"].append(row)
    result["all_actual_paired_series_exactly_equal"] = all(s["direct_series_array_equal"] for c in result["currents"] for s in c["signals"].values())
    result["paired_ci_zero_for_all_signals"] = all(all(abs(x) == 0.0 for x in s["paired_95pct_CI"]) for c in result["currents"] for s in c["signals"].values())
    result["autocorrelation_note"] = "Block adequacy is reported per reference trajectory; fewer than 10 blocks is inconclusive, not a failure of physical equivalence."
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=pathlib.Path); p.add_argument("--output", type=pathlib.Path, required=True)
    a = p.parse_args(argv)
    try:
        data = analyze(a.receipt)
    except Exception as exc:
        data = {"schema": "tdgl-production-statistics-v1", "valid": False, "failures": [str(exc)]}
    a.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": data.get("valid"), "currents": len(data.get("currents", [])), "output": str(a.output)}))
    return 0 if data.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
