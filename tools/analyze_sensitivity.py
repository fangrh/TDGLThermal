#!/usr/bin/env python3
"""Extract reviewer-facing NDR and thermal metrics from TDGL HDF5 output."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tools.generate_sensitivity_configs import read_yaml_scalar


@dataclass(frozen=True)
class NDRMetrics:
    present: bool
    onset_current: float | None
    minimum_slope: float | None
    ndr_area: float


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    onset_relative_difference: float | None
    area_relative_difference: float | None


@dataclass(frozen=True)
class RunSeries:
    current: np.ndarray
    voltage: np.ndarray
    suspended_temperature_mean: np.ndarray
    suspended_temperature_peak: np.ndarray


def _validate_curve(current: np.ndarray, voltage: np.ndarray) -> None:
    if current.ndim != 1 or voltage.ndim != 1 or len(current) != len(voltage):
        raise ValueError("current and voltage must be equal-length one-dimensional arrays")
    if len(current) < 2:
        raise ValueError("at least two current points are required")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(voltage)):
        raise ValueError("current and voltage must be finite")
    if not np.all(np.diff(current) < 0.0):
        raise ValueError("return-branch current must be strictly descending")


def _contiguous_true_groups(mask: np.ndarray) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            groups.append((start, index))
            start = None
    if start is not None:
        groups.append((start, len(mask)))
    return groups


def compute_ndr_metrics(current: np.ndarray, voltage: np.ndarray) -> NDRMetrics:
    current = np.asarray(current, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    _validate_curve(current, voltage)
    delta_current = np.diff(current)
    delta_voltage = np.diff(voltage)
    slopes = delta_voltage / delta_current
    groups = [group for group in _contiguous_true_groups(slopes < 0.0) if group[1] - group[0] >= 2]
    if not groups:
        return NDRMetrics(False, None, None, 0.0)

    start, stop = groups[0]
    ndr_area = float(
        np.sum(np.maximum(delta_voltage[start:stop], 0.0) * np.abs(delta_current[start:stop]))
    )
    return NDRMetrics(
        present=True,
        onset_current=float(current[start]),
        minimum_slope=float(np.min(slopes[start:stop])),
        ndr_area=ndr_area,
    )


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), np.finfo(float).eps)


def evaluate_gate(legacy: NDRMetrics, corrected: NDRMetrics) -> GateDecision:
    if (
        not legacy.present
        or not corrected.present
        or legacy.onset_current is None
        or corrected.onset_current is None
    ):
        return GateDecision(False, None, None)
    onset_delta = relative_difference(legacy.onset_current, corrected.onset_current)
    area_delta = relative_difference(legacy.ndr_area, corrected.ndr_area)
    return GateDecision(
        passed=onset_delta <= 0.05 and area_delta <= 0.05,
        onset_relative_difference=onset_delta,
        area_relative_difference=area_delta,
    )


def _read_required(handle: h5py.File, key: str) -> np.ndarray:
    if key not in handle:
        raise ValueError(f"missing HDF5 dataset: {key}")
    return np.asarray(handle[key])


def load_run(run_dir: Path) -> RunSeries:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"missing run config: {config_path}")
    config = config_path.read_text(encoding="utf-8")
    nx = int(read_yaml_scalar(config, "grid", "Nx"))
    ny = int(read_yaml_scalar(config, "grid", "Ny"))
    hx = read_yaml_scalar(config, "grid", "hx")
    holewidth = read_yaml_scalar(config, "thermal", "holewidth")
    x = np.linspace(-nx * hx / 2.0, nx * hx / 2.0, nx + 1)
    suspended_x = np.abs(x) < holewidth / 2.0
    if not np.any(suspended_x):
        raise ValueError("suspended-region mask contains no grid nodes")

    rows: list[tuple[float, float, float, float]] = []
    for path in run_dir.rglob("current_Je*_down.h5"):
        with h5py.File(path, "r") as handle:
            je = float(_read_required(handle, "Je").reshape(()))
            voltage = _read_required(handle, "V").astype(float).ravel()
            temperature = _read_required(handle, "T").astype(float)
            if voltage.size == 0:
                raise ValueError(f"empty HDF5 voltage dataset: {path}")
            if temperature.ndim != 3 or temperature.shape[1:] != (ny + 1, nx + 1):
                raise ValueError(f"unexpected HDF5 temperature shape: {path}")
            interior = temperature[:, 1:ny, :][:, :, suspended_x]
            if interior.size == 0:
                raise ValueError(f"empty suspended temperature selection: {path}")
            values = np.concatenate((voltage, interior.ravel()))
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite HDF5 values: {path}")
            rows.append(
                (
                    je,
                    float(np.mean(voltage)),
                    float(np.mean(interior)),
                    float(np.max(interior)),
                )
            )
    if not rows:
        raise ValueError(f"no return-branch HDF5 files found under {run_dir}")
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"duplicate return-branch currents under {run_dir}")
    rows.sort(key=lambda row: row[0], reverse=True)
    array = np.asarray(rows, dtype=float)
    return RunSeries(array[:, 0], array[:, 1], array[:, 2], array[:, 3])


def _metrics_dict(series: RunSeries) -> dict[str, object]:
    metrics = compute_ndr_metrics(series.current, series.voltage)
    return {
        **asdict(metrics),
        "suspended_temperature_mean": float(np.mean(series.suspended_temperature_mean)),
        "suspended_temperature_peak": float(np.max(series.suspended_temperature_peak)),
        "current_points": int(len(series.current)),
    }


def _plot_series(named: list[tuple[str, RunSeries]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for name, series in named:
        axis.plot(series.current, series.voltage, marker="o", markersize=2, label=name)
    axis.set_xlabel("Dimensionless current density J")
    axis.set_ylabel("Dimensionless voltage V")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def analyze_gate(legacy_dir: Path, corrected_dir: Path, output_dir: Path) -> GateDecision:
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy = load_run(legacy_dir)
    corrected = load_run(corrected_dir)
    legacy_metrics = compute_ndr_metrics(legacy.current, legacy.voltage)
    corrected_metrics = compute_ndr_metrics(corrected.current, corrected.voltage)
    decision = evaluate_gate(legacy_metrics, corrected_metrics)
    report = {
        "legacy": _metrics_dict(legacy),
        "corrected": _metrics_dict(corrected),
        "gate": asdict(decision),
    }
    (output_dir / "gate_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _plot_series(
        [("legacy", legacy), ("corrected full curl-curl", corrected)],
        output_dir / "gate_iv.pdf",
    )
    return decision


def analyze_matrix(root: Path, output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    plotted: list[tuple[str, RunSeries]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (run_dir / "config.yaml").is_file():
            continue
        series = load_run(run_dir)
        rows.append({"run_name": run_dir.name, **_metrics_dict(series)})
        plotted.append((run_dir.name, series))
    if not rows:
        raise ValueError(f"no completed matrix runs found under {root}")
    fieldnames = list(rows[0].keys())
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _plot_series(plotted, output_dir / "sensitivity_iv.pdf")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--legacy", type=Path, required=True)
    gate.add_argument("--corrected", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--root", type=Path, required=True)
    matrix.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "gate":
        decision = analyze_gate(args.legacy, args.corrected, args.output)
        print(json.dumps(asdict(decision), indent=2))
    else:
        rows = analyze_matrix(args.root, args.output)
        print(f"Analyzed {len(rows)} matrix runs")


if __name__ == "__main__":
    main()
