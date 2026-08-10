#!/usr/bin/env python3
"""Package immutable TDGL reviewer-sensitivity Slurm bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


CURRENT_SOLVER_BLOB = "3a89981e3ef4b3fa241d0a19645715cecd0e9228"
LEGACY_SOLVER_BLOB = "93a34dd8f50a76d4ef014d37d541ac36de6955c3"
SOLVER_SOURCE_COMMIT = "1969372be4da34eb73462eef66f823a57e64b622"
REQUIRED_BUNDLE_FILES = {
    "animation_data_utils.jl",
    "config.yaml",
    "current_sweep_thermal_nofastscan.jl",
    "run_metadata.json",
    "submit.sbatch",
}


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def canonical_solver_bytes(path: Path) -> bytes:
    """Return platform-independent LF bytes for solver provenance and upload."""
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one solver fragment, found {count}: {old.strip()}")
    return text.replace(old, new, 1)


def reconstruct_legacy_solver(current_text: str) -> str:
    text = current_text.replace("\r\n", "\n")
    text = _replace_once(
        text,
        "            # The TDGL equation contains -curl curl A = laplacian(A) - grad(div A).\n",
        "",
    )
    text = _replace_once(
        text,
        "            dxy_Ay = ((Ay[i+1,j] - Ay[i,j]) - (Ay[i+1,j-1] - Ay[i,j-1])) / (p.hx * p.hy)\n",
        "",
    )
    text = _replace_once(
        text,
        "            dAx[i,j] = (Jsx[i,j] + kappa2 * (d2Ax_dy2 - dxy_Ay)) / sigma_local\n",
        "            dAx[i,j] = (Jsx[i,j] + kappa2 * d2Ax_dy2) / sigma_local\n",
    )
    text = _replace_once(
        text,
        "            dxy_Ax = ((Ax[i,j+1] - Ax[i-1,j+1]) - (Ax[i,j] - Ax[i-1,j])) / (p.hx * p.hy)\n",
        "",
    )
    text = _replace_once(
        text,
        "            dAy[i,j] = (Jsy[i,j] + kappa2 * (d2Ay_dx2 - dxy_Ax)) / sigma_local\n",
        "            dAy[i,j] = (Jsy[i,j] + kappa2 * d2Ay_dx2) / sigma_local\n",
    )
    if git_blob_sha1(text.encode("utf-8")) != LEGACY_SOLVER_BLOB:
        raise ValueError("reconstructed legacy solver does not match archived blob")
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(bundle: Path) -> None:
    paths = sorted(
        path for path in bundle.iterdir() if path.is_file() and path.name != "manifest.sha256"
    )
    lines = [f"{_sha256(path)}  {path.name}" for path in paths]
    (bundle / "manifest.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def verify_bundle(bundle: Path) -> bool:
    try:
        names = {path.name for path in bundle.iterdir() if path.is_file()}
        if not REQUIRED_BUNDLE_FILES.issubset(names) or "manifest.sha256" not in names:
            return False
        lines = (bundle / "manifest.sha256").read_text(encoding="utf-8").splitlines()
        if not lines:
            return False
        for line in lines:
            expected, relative = line.split("  ", 1)
            path = bundle / relative
            if not path.is_file() or _sha256(path) != expected:
                return False
        return True
    except (OSError, ValueError):
        return False


def _prepare_bundle(
    root: Path,
    output_dir: Path,
    name: str,
    config_path: Path,
    solver_bytes: bytes,
    solver_role: str,
    changes: list[dict[str, object]],
    stage: str,
) -> Path:
    bundle = output_dir / name
    if bundle.exists() and any(bundle.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty bundle: {bundle}")
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, bundle / "config.yaml")
    shutil.copyfile(root / "animation_data_utils.jl", bundle / "animation_data_utils.jl")
    shutil.copyfile(
        root / "studies/reviewer_sensitivity/submit.sbatch", bundle / "submit.sbatch"
    )
    (bundle / "current_sweep_thermal_nofastscan.jl").write_bytes(solver_bytes)
    metadata = {
        "run_name": name,
        "stage": stage,
        "changes": changes,
        "solver_role": solver_role,
        "solver_git_blob": git_blob_sha1(solver_bytes),
        "solver_source_commit": SOLVER_SOURCE_COMMIT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (bundle / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    _write_manifest(bundle)
    if not verify_bundle(bundle):
        raise ValueError(f"bundle verification failed: {bundle}")
    return bundle


def package_runs(
    root: Path,
    output_dir: Path,
    stage: str,
    configs_dir: Path | None = None,
) -> list[Path]:
    root = root.resolve()
    current_bytes = canonical_solver_bytes(
        root / "current_sweep_thermal_nofastscan.jl"
    )
    if git_blob_sha1(current_bytes) != CURRENT_SOLVER_BLOB:
        raise ValueError("current solver blob does not match approved corrected solver")
    current_text = current_bytes.decode("utf-8")
    legacy_bytes = reconstruct_legacy_solver(current_text).encode("utf-8")
    baseline = root / "studies/reviewer_sensitivity/baseline.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)

    if stage == "gate":
        definitions = [
            ("baseline-legacy-t018", baseline, legacy_bytes, "legacy", []),
            ("baseline-fullcurl-t018", baseline, current_bytes, "corrected", []),
        ]
    elif stage in {"matrix", "temperature-gate"}:
        if configs_dir is None:
            raise ValueError("configs_dir is required for matrix packaging")
        manifest = json.loads(
            (configs_dir / "generated_matrix.json").read_text(encoding="utf-8")
        )
        definitions = []
        for run in manifest["runs"]:
            if run["name"] == manifest["baseline"]:
                continue
            definitions.append(
                (
                    run["name"],
                    configs_dir / run["config"],
                    current_bytes,
                    "corrected",
                    run["changes"],
                )
            )
    else:
        raise ValueError(f"unsupported stage: {stage}")

    return [
        _prepare_bundle(
            root,
            output_dir,
            name,
            config,
            solver,
            solver_role,
            changes,
            stage,
        )
        for name, config, solver, solver_role, changes in definitions
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gate", "temperature-gate", "matrix"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()

    if args.verify is not None:
        if not verify_bundle(args.verify):
            raise SystemExit(f"Bundle verification failed: {args.verify}")
        print(f"Bundle verified: {args.verify}")
        return
    if args.stage is None or args.output is None:
        parser.error("--stage and --output are required unless --verify is used")

    root = Path(__file__).resolve().parents[1]
    bundles = package_runs(root, args.output, args.stage, args.configs)
    print(f"Packaged {len(bundles)} {args.stage} runs in {args.output}")


if __name__ == "__main__":
    main()
