#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

julia "$ROOT_DIR/test/distributed_continuation_definition.jl"

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

mkdir -p "$WORK_DIR/input"
touch "$WORK_DIR/input/current_Je0.1000_up.h5"
touch "$WORK_DIR/input/current_Je0.2000_up.h5"
touch "$WORK_DIR/input/current_Je0.3000_up.h5"

cat > "$WORK_DIR/config.yaml" <<EOF
# TDGL + Thermal No-FastScan Configuration

sweep:
  Jpeak: 0.3
  n_current_points: 3
  input_folder: input
  stable_time: 100.0
  dt_snapshots: 1.0
  skip_ratio: 0.5
  run_up: true
  run_down: false

outputs:
  generate_mp4: false

continuation_scan:
  start: 0.11
  end: 0.28
  stable: 50
  ramptime: 5
EOF

(
    cd "$WORK_DIR"
    julia "$ROOT_DIR/current_sweep_thermal_continue.jl" config.yaml --dry-run > dry_run.log
)

grep -q "^DRY_RUN selected_direction=up$" "$WORK_DIR/dry_run.log"
grep -q "^DRY_RUN selected_start=0.1000$" "$WORK_DIR/dry_run.log"
grep -q "^DRY_RUN targets=0.1000,0.2000$" "$WORK_DIR/dry_run.log"
grep -q "^OUTPUT_DIR:sweep_nofastscan_" "$WORK_DIR/dry_run.log"

mkdir -p "$WORK_DIR/down_input"
touch "$WORK_DIR/down_input/current_Je0.3000_down.h5"
touch "$WORK_DIR/down_input/current_Je0.2000_down.h5"
touch "$WORK_DIR/down_input/current_Je0.1000_down.h5"

cat > "$WORK_DIR/down_config.yaml" <<EOF
# TDGL + Thermal No-FastScan Configuration

sweep:
  Jpeak: 0.3
  n_current_points: 3
  input_folder: down_input
  stable_time: 100.0
  dt_snapshots: 1.0
  skip_ratio: 0.5
  run_up: false
  run_down: true

outputs:
  generate_mp4: false

continuation_scan:
  start: 0.29
  end: 0.11
  stable: 50
  ramptime: 5
EOF

(
    cd "$WORK_DIR"
    julia "$ROOT_DIR/current_sweep_thermal_continue.jl" down_config.yaml --dry-run > down_dry_run.log
)

grep -q "^DRY_RUN selected_direction=down$" "$WORK_DIR/down_dry_run.log"
grep -q "^DRY_RUN selected_start=0.2000$" "$WORK_DIR/down_dry_run.log"
grep -q "^DRY_RUN targets=0.2000$" "$WORK_DIR/down_dry_run.log"

mkdir -p "$WORK_DIR/missing_direction_input"
touch "$WORK_DIR/missing_direction_input/current_Je0.1000_up.h5"
touch "$WORK_DIR/missing_direction_input/current_Je0.2000_down.h5"

cat > "$WORK_DIR/missing_direction_config.yaml" <<EOF
# TDGL + Thermal No-FastScan Configuration

sweep:
  Jpeak: 0.1
  n_current_points: 1
  input_folder: missing_direction_input
  stable_time: 100.0
  dt_snapshots: 1.0
  skip_ratio: 0.5
  run_up: true
  run_down: false

outputs:
  generate_mp4: false

continuation_scan:
  start: 0.20
  end: 0.05
  stable: 50
  ramptime: 5
EOF

(
    cd "$WORK_DIR"
    julia "$ROOT_DIR/current_sweep_thermal_continue.jl" missing_direction_config.yaml --dry-run > missing_direction.log
)
grep -q "^DRY_RUN selected_direction=down$" "$WORK_DIR/missing_direction.log"
grep -q "^DRY_RUN selected_start=0.2000$" "$WORK_DIR/missing_direction.log"
grep -q "^DRY_RUN selected_source_direction=down$" "$WORK_DIR/missing_direction.log"
grep -q "^DRY_RUN targets=0.2000,0.1000$" "$WORK_DIR/missing_direction.log"

mkdir -p "$WORK_DIR/runtime_input"
julia -e '
using HDF5
h5open(ARGS[1], "w") do file
    file["psi_real"] = reshape(ones(Float64, 25), 5, 5, 1)
    file["psi_imag"] = reshape(zeros(Float64, 25), 5, 5, 1)
    file["T"] = reshape(zeros(Float64, 25), 5, 5, 1)
    file["Ax"] = reshape(zeros(Float64, 20), 4, 5, 1)
    file["Ay"] = reshape(zeros(Float64, 20), 5, 4, 1)
end
' "$WORK_DIR/runtime_input/current_Je0.1000_up.h5"

cat > "$WORK_DIR/runtime_config.yaml" <<EOF
# TDGL + Thermal No-FastScan Configuration

grid:
  Nx: 4
  Ny: 4
  hx: 0.5
  hy: 0.5

sweep:
  Jpeak: 0.1
  n_current_points: 1
  input_folder: runtime_input
  stable_time: 0.05
  dt_snapshots: 0.05
  skip_ratio: 0.0
  run_up: true
  run_down: false

outputs:
  generate_mp4: false

continuation_scan:
  start: 0.10
  end: 0.11
  stable: 0.05
  ramptime: 0.01
EOF

(
    cd "$WORK_DIR"
    julia -p 1 "$ROOT_DIR/current_sweep_thermal_continue.jl" runtime_config.yaml > runtime.log
)

OUTPUT_DIR=$(awk -F: '/^OUTPUT_DIR:/{print $2; exit}' "$WORK_DIR/runtime.log")
test -n "$OUTPUT_DIR"
test -d "$WORK_DIR/$OUTPUT_DIR"
test -f "$WORK_DIR/$OUTPUT_DIR/config.yaml"
test -f "$WORK_DIR/$OUTPUT_DIR/current_Je0.1000_up.h5"

mkdir -p "$WORK_DIR/runtime_input_multi"
julia -e '
using HDF5
for path in ARGS
    h5open(path, "w") do file
        file["psi_real"] = reshape(ones(Float64, 25), 5, 5, 1)
        file["psi_imag"] = reshape(zeros(Float64, 25), 5, 5, 1)
        file["T"] = reshape(zeros(Float64, 25), 5, 5, 1)
        file["Ax"] = reshape(zeros(Float64, 20), 4, 5, 1)
        file["Ay"] = reshape(zeros(Float64, 20), 5, 4, 1)
    end
end
' \
"$WORK_DIR/runtime_input_multi/current_Je0.1000_up.h5" \
"$WORK_DIR/runtime_input_multi/current_Je0.2000_up.h5"

cat > "$WORK_DIR/runtime_multi_config.yaml" <<EOF
# TDGL + Thermal No-FastScan Configuration

grid:
  Nx: 4
  Ny: 4
  hx: 0.5
  hy: 0.5

sweep:
  Jpeak: 0.2
  n_current_points: 2
  input_folder: runtime_input_multi
  stable_time: 0.05
  dt_snapshots: 0.05
  skip_ratio: 0.0
  run_up: true
  run_down: false

outputs:
  generate_mp4: false

continuation_scan:
  start: 0.10
  end: 0.20
  stable: 0.05
  ramptime: 0.01
EOF

(
    cd "$WORK_DIR"
    julia -p 1 "$ROOT_DIR/current_sweep_thermal_continue.jl" runtime_multi_config.yaml > runtime_multi.log
)

MULTI_OUTPUT_DIR=$(awk -F: '/^OUTPUT_DIR:/{print $2; exit}' "$WORK_DIR/runtime_multi.log")
test -n "$MULTI_OUTPUT_DIR"
test -f "$WORK_DIR/$MULTI_OUTPUT_DIR/current_Je0.1000_up.h5"
test -f "$WORK_DIR/$MULTI_OUTPUT_DIR/current_Je0.2000_up.h5"
