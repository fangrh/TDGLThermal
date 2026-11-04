"""Illustrative final field maps for selected full-curl NDR endpoints.

The script reads local copies of endpoint chain/HDF5 artifacts; it performs no
SSH or simulation.  It intentionally labels the maps as selected snapshots,
not as a population or stationarity demonstration.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = (
    ("endpoint_a_seed2026110412", "Ja, seed 2026110412"),
    ("endpoint_b_seed2026110412", "Jb, seed 2026110412 (low-voltage branch)"),
    ("endpoint_b_seed2026110413", "Jb, seed 2026110413 (high-voltage branch)"),
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load_row(root: Path, analysis: dict, job_id: str):
    chain_path = next(root.glob(f"{job_id}/segment10/chain.json"), None)
    if chain_path is None:
        raise FileNotFoundError(f"missing {job_id}/segment10/chain.json under {root}")
    chain = json.loads(chain_path.read_text(encoding="utf-8-sig"))
    recorded_h5 = str(chain["manifest"]["output_path"])
    remote_h5 = Path(recorded_h5)
    local_h5 = chain_path.parent / remote_h5.name
    if not local_h5.is_file():
        local_h5 = remote_h5
    if not local_h5.is_file():
        raise FileNotFoundError(f"HDF5 not found locally for {job_id}: {remote_h5}")
    seed = int(chain["seed"]); endpoint = "endpoint_a" if "endpoint_a" in job_id else "endpoint_b"
    candidates = [x for x in analysis["late_trajectory"] if x.get("id") == endpoint and int(x.get("seed")) == seed and int(x.get("segment")) == 10]
    if len(candidates) != 1:
        raise ValueError(f"expected one segment-10 analysis row for {job_id}, got {len(candidates)}")
    with h5py.File(local_h5, "r") as f:
        times = np.asarray(f["times"]); t = float(times[-1])
        psi2 = np.asarray(f["psi_real"][-1]) ** 2 + np.asarray(f["psi_imag"][-1]) ** 2
        temp = np.asarray(f["T"][-1])
        ax = np.asarray(f["Ax"][-1]); ay = np.asarray(f["Ay"][-1])
        # Stored arrays are [y,x] at the final snapshot; staggered curl gives [y,x].
        bz = (ay[:, 1:] - ay[:, :-1]) / 0.5 - (ax[1:, :] - ax[:-1, :]) / 0.5
        shapes = {k: tuple(f[k].shape) for k in ("psi_real", "T", "Ax", "Ay")}
    digest = sha256(local_h5)
    if digest != chain["manifest"]["output_hash"]:
        raise ValueError(f"HDF5 hash mismatch for {job_id}")
    return dict(chain_path=str(chain_path), chain=chain, h5_path=str(local_h5), remote_h5=recorded_h5,
                h5_sha256=digest, time=t, psi2=psi2, temp=temp, bz=bz,
                voltage=float(candidates[0]["V"]), shapes=shapes)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-root", required=True, type=Path)
    ap.add_argument("--endpoint-analysis", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    analysis = json.loads(args.endpoint_analysis.read_text(encoding="utf-8-sig"))
    rows = [load_row(args.endpoint_root, analysis, job_id) for job_id, _ in ROWS]
    psi_max = max(1.0, max(float(np.nanmax(r["psi2"])) for r in rows))
    tmin = min(float(np.nanmin(r["temp"])) for r in rows); tmax = max(float(np.nanmax(r["temp"])) for r in rows)
    bmax = max(float(np.nanmax(np.abs(r["bz"]))) for r in rows)
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.4), constrained_layout=True)
    fields = [("psi2", r"$|\psi|^2$", "viridis", (0, psi_max)), ("temp", "T", "inferno", (tmin, tmax)), ("bz", r"$B_z$", "coolwarm", (-bmax, bmax))]
    ims = []
    for i, ((job_id, row_label), row) in enumerate(zip(ROWS, rows)):
        for j, (key, title, cmap, limits) in enumerate(fields):
            ax = axes[i, j]; data = row[key]
            im = ax.imshow(data, origin="lower", extent=(-15, 15, -7.5, 7.5), aspect="equal", cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="nearest")
            ims.append(im)
            if i == 0: ax.set_title(title, fontsize=13)
            if j == 0: ax.set_ylabel(("A · seed 12", "B · seed 12 (low V)", "B · seed 13 (high V)")[i] + "\ny (model units)")
            else: ax.set_ylabel("y (model units)")
            ax.set_xlabel("x (model units)")
            ax.text(0.02, 0.97, f"$J_e$={row['chain']['Je']:.6f}\n$\\overline{{V}}$={row['voltage']:.4f}", transform=ax.transAxes, va="top", ha="left", color="white", fontsize=8, bbox=dict(facecolor="black", alpha=0.45, pad=2, edgecolor="none"))
    for j, im in enumerate(ims[:3]):
        cbar = fig.colorbar(im, ax=axes[:, j], fraction=0.04, pad=0.015, shrink=.85); cbar.set_label(fields[j][1])
    fig.suptitle("Selected full-curl endpoint fields at cumulative 50,000 $\\tau_0$", fontsize=15)
    fig.supxlabel("Selected final snapshots; voltage labels average 48000–50000τ.\nPopulation statistics and stationarity tests are reported separately.", fontsize=9)
    args.out.mkdir(parents=True, exist_ok=True)
    png = args.out / "fields.png"; pdf = args.out / "fields.pdf"; fig.savefig(png, dpi=220); fig.savefig(pdf); plt.close(fig)
    provenance = {"script": str(Path(__file__).resolve()), "analysis_path": str(args.endpoint_analysis.resolve()), "analysis_sha256": sha256(args.endpoint_analysis), "endpoint_root": str(args.endpoint_root.resolve()), "selected_local_time_tau0": {r["chain"]["id"]: r["time"] for r in rows}, "selected_cumulative_time_tau0": {r["chain"]["id"]: r["chain"]["cumulative_end"] for r in rows}, "rows": [{"id": r["chain"]["id"], "chain_path": str(Path(r["chain_path"]).resolve()), "remote_hdf5": r["remote_h5"], "local_hdf5": str(Path(r["h5_path"]).resolve()), "hdf5_sha256": r["h5_sha256"], "manifest_output_sha256": r["chain"]["manifest"].get("output_hash"), "Je": r["chain"]["Je"], "seed": r["chain"]["seed"], "window_mean_V": r["voltage"], "dataset_shapes": r["shapes"]} for r in rows], "scales": {"psi2": [0, psi_max], "T": [tmin, tmax], "Bz": [-bmax, bmax]}, "note": "Illustrative selected snapshots only; no representative-population or steady-NDR claim."}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
