#!/usr/bin/env python3
"""Plot receipt-verified spatial state-response fields for one predeclared seed."""
from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 2026110411
LABELS = ("inner1", "inner2")
ARMS = ("minus", "sham", "plus")
IDS = {f"{l}_{a}_seed{SEED}" for l in LABELS for a in ARMS}


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _require_analysis(path: Path) -> dict:
    d = _read(path)
    required = ("schema", "complete", "artifact_validation", "finite_saved_fields_validation", "conditioning_identity_checked")
    if d.get("schema") != "tdgl-full-curl-local-response-analysis-v1": raise ValueError("analysis schema mismatch")
    if any(d.get(k) is not True for k in required[1:]): raise ValueError("analysis is not fully verified")
    if d.get("errors") != []: raise ValueError("analysis contains errors")
    return d


def _chain_for(root: Path, jid: str, analysis: dict) -> tuple[Path, dict, dict]:
    found = list(root.glob(f"{jid}/step5_Je*/chain.json"))
    if len(found) != 1: raise ValueError(f"expected exactly one step-5 receipt for {jid}, found {len(found)}")
    chain = found[0]; entry = _read(chain); m = entry.get("manifest", {})
    declared = analysis.get("provenance", {}).get("chain_receipt_sha256", {})
    actual = _sha(chain)
    allowed = declared.get(str(chain.resolve()), declared.get(str(chain)))
    if allowed != actual: raise ValueError(f"chain receipt provenance mismatch: {jid}")
    if entry.get("id") != jid or entry.get("step") != 5 or entry.get("seed") != SEED:
        raise ValueError(f"chain identity mismatch: {jid}")
    if not math.isclose(float(entry.get("cumulative_end", math.nan)), 75000., rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"cumulative endpoint mismatch: {jid}")
    h5 = Path(str(m.get("output_path", "")))
    if not h5.is_file() or _sha(h5) != m.get("output_hash"): raise ValueError(f"HDF5 hash mismatch: {jid}")
    return chain, entry, m


def _load_field(m: dict, jid: str) -> tuple[np.ndarray, np.ndarray, float, dict]:
    h5 = Path(str(m["output_path"]))
    with h5py.File(h5, "r") as f:
        for name in ("times", "psi_real", "psi_imag", "T"):
            if name not in f: raise ValueError(f"missing dataset {name}: {jid}")
        times = np.asarray(f["times"])
        psi_r, psi_i, temp = (np.asarray(f[x][-1, :, :]) for x in ("psi_real", "psi_imag", "T"))
        if times.size == 0 or not math.isclose(float(times[-1]), 5000., rel_tol=0, abs_tol=1e-9): raise ValueError(f"time endpoint mismatch: {jid}")
        if psi_r.shape != (31, 61) or psi_i.shape != (31, 61) or temp.shape != (31, 61): raise ValueError(f"field shape mismatch: {jid}")
        if not all(np.all(np.isfinite(x)) for x in (psi_r, psi_i, temp)): raise ValueError(f"nonfinite field: {jid}")
    return psi_r * psi_r + psi_i * psi_i, temp, float(times[-1]), {"shape": list(psi_r.shape), "h5_sha256": _sha(h5)}


def plot(analysis_path: Path, campaign_root: Path, out: Path) -> dict:
    analysis = _require_analysis(analysis_path)
    rows = {}
    for jid in sorted(IDS):
        chain, entry, manifest = _chain_for(campaign_root, jid, analysis)
        if manifest.get("hx") != .5 or manifest.get("hy") != .5: raise ValueError(f"grid spacing mismatch: {jid}")
        psi2, temp, local_time, info = _load_field(manifest, jid)
        rows[jid] = {"chain": chain, "entry": entry, "manifest": manifest, "psi2": psi2, "T": temp, "local_time": local_time, **info}
    psi_min, psi_max = min(float(x["psi2"].min()) for x in rows.values()), max(float(x["psi2"].max()) for x in rows.values())
    t_min, t_max = min(float(x["T"].min()) for x in rows.values()), max(float(x["T"].max()) for x in rows.values())
    if not (math.isfinite(psi_min) and math.isfinite(psi_max) and math.isfinite(t_min) and math.isfinite(t_max)): raise ValueError("nonfinite plot range")
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), constrained_layout=True)
    extent = (0., 30., 0., 15.)
    ims = []
    for col, arm in enumerate(ARMS):
        for source_row, label in enumerate(LABELS):
            x = rows[f"{label}_{arm}_seed{SEED}"]
            for row, data, lo, hi, title in ((source_row * 2, x["psi2"], psi_min, psi_max, rf"{label}: $|\psi|^2$, {arm}"),
                                               (source_row * 2 + 1, x["T"], t_min, t_max, rf"{label}: $T$, {arm}")):
                im = axes[row, col].imshow(data, origin="lower", extent=extent, aspect="auto", vmin=lo, vmax=hi, cmap="viridis")
                axes[row, col].set(title=title + f" · Je={x['manifest']['Je']:.6f}", xlabel="x (native units)", ylabel="y (native units)")
                ims.append((row, im))
    for row, label in ((0, r"$|\psi|^2$"), (1, "T"), (2, r"$|\psi|^2$"), (3, "T")):
        group = [ims[i][1] for i in range(len(ims)) if ims[i][0] == row]
        fig.colorbar(group[0], ax=axes[row, :], label=label, shrink=.85)
    fig.suptitle("Illustrative local current-state response: seed 2026110411\nSelected snapshots at cumulative time 75000τ")
    fig.text(.5, .005, "Illustrative reused-seed snapshots; not population-representative and not proof of a stationary state.", ha="center", fontsize=9)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "state_response_fields.png", dpi=180)
    fig.savefig(out / "state_response_fields.pdf")
    plt.close(fig)
    provenance = {"analysis_path": str(analysis_path.resolve()), "analysis_sha256": _sha(analysis_path), "script_sha256": _sha(Path(__file__)),
                  "seed": SEED, "cumulative_time": 75000, "extent": [0, 30, 0, 15], "psi2_range": [psi_min, psi_max], "T_range": [t_min, t_max],
                  "selected": {jid: {"chain_path": str(v["chain"].resolve()), "chain_sha256": _sha(v["chain"]), "hdf5_path": str(Path(v["manifest"]["output_path"]).resolve()),
                                     "hdf5_sha256": v["h5_sha256"], "local_time": v["local_time"], "source_Je": analysis['protocol']['source_currents'][jid.split('_')[0]], "target_Je": v["entry"].get("Je")} for jid, v in rows.items()},
                  "scope": "illustrative verified spatial snapshots; no population or stationary-state claim"}
    (out / "field_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--analysis", required=True); p.add_argument("--campaign-root", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); plot(Path(a.analysis).resolve(), Path(a.campaign_root).resolve(), Path(a.out).resolve())


if __name__ == "__main__": main()
