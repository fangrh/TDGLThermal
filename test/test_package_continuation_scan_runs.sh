#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

mkdir -p "$WORK_DIR/bin"
cat > "$WORK_DIR/bin/sbatch" <<EOF
#!/bin/bash
set -euo pipefail
printf '%s\n' "\$*" >> "$WORK_DIR/sbatch.log"
EOF
chmod +x "$WORK_DIR/bin/sbatch"
export PATH="$WORK_DIR/bin:$PATH"

"$ROOT_DIR/package_continuation_scan_runs.sh" \
  --input_folder "$ROOT_DIR/test/fixtures/package_input/changeT1" \
  --output_folder "$WORK_DIR/output" \
  --start 0.1 \
  --end 0.2 \
  --stable 5000 \
  --ramptime 50

test -d "$WORK_DIR/output/runA"
test -d "$WORK_DIR/output/runB"
test -L "$WORK_DIR/output/input_folder"
test -f "$WORK_DIR/output/runA/config.yaml"
test -f "$WORK_DIR/output/runA/current_sweep_thermal_continue.jl"
test -f "$WORK_DIR/output/runA/current_sweep_thermal_nofastscan.jl"
test -f "$WORK_DIR/output/runA/animation_data_utils.jl"
test -f "$WORK_DIR/output/runA/submit_scan.sh"
test -x "$WORK_DIR/output/runA/run.sh"
test -x "$WORK_DIR/output/runB/run.sh"
test -x "$WORK_DIR/output/run_all.sh"
grep -q "input_folder: ../input_folder/runA/source_data" "$WORK_DIR/output/runA/config.yaml"
grep -q "stable_time: 5000" "$WORK_DIR/output/runA/config.yaml"
grep -q "^continuation_scan:$" "$WORK_DIR/output/runA/config.yaml"
grep -q "start: 0.1" "$WORK_DIR/output/runA/config.yaml"
grep -q "end: 0.2" "$WORK_DIR/output/runA/config.yaml"
grep -q "stable: 5000" "$WORK_DIR/output/runA/config.yaml"
grep -q "ramptime: 50" "$WORK_DIR/output/runA/config.yaml"
test "$(grep -c '^continuation_scan:$' "$WORK_DIR/output/runA/config.yaml")" -eq 1
grep -q '^julia -p 31 current_sweep_thermal_continue.jl "\$@"$' "$WORK_DIR/output/runA/submit_scan.sh"
grep -q '^sbatch submit_scan.sh config.yaml$' "$WORK_DIR/output/runA/run.sh"
grep -q 'bash "$ROOT_DIR/runA/run.sh"' "$WORK_DIR/output/run_all.sh"
grep -q 'bash "$ROOT_DIR/runB/run.sh"' "$WORK_DIR/output/run_all.sh"
test -f "$WORK_DIR/sbatch.log"
test "$(wc -l < "$WORK_DIR/sbatch.log")" -eq 2
grep -q "^submit_scan.sh config.yaml$" "$WORK_DIR/sbatch.log"

mkdir -p "$WORK_DIR/chain_input/runC/sweep_nofastscan_20260323_010203"
cat > "$WORK_DIR/chain_input/runC/config.yaml" <<EOF
# TDGL + Thermal No-FastScan Configuration

sweep:
  Jpeak: 0.3
  n_current_points: 3
  input_folder: old_input
  stable_time: 10.0
  dt_snapshots: 1.0
  skip_ratio: 0.2
  run_up: true
  run_down: false

outputs:
  generate_mp4: false

continuation_scan:
  start: 0.05
  end: 0.15
  stable: 20
  ramptime: 2
EOF
printf 'placeholder\n' > "$WORK_DIR/chain_input/runC/sweep_nofastscan_20260323_010203/current_Je0.1000_up.h5"

"$ROOT_DIR/package_continuation_scan_runs.sh" \
  --input_folder "$WORK_DIR/chain_input" \
  --output_folder "$WORK_DIR/chain_output" \
  --start 0.1 \
  --end 0.2 \
  --stable 30 \
  --ramptime 4

grep -q "input_folder: ../input_folder/runC/sweep_nofastscan_20260323_010203" "$WORK_DIR/chain_output/runC/config.yaml"
test "$(grep -c '^continuation_scan:$' "$WORK_DIR/chain_output/runC/config.yaml")" -eq 1
grep -q "start: 0.1" "$WORK_DIR/chain_output/runC/config.yaml"
grep -q "end: 0.2" "$WORK_DIR/chain_output/runC/config.yaml"
grep -q "stable: 30" "$WORK_DIR/chain_output/runC/config.yaml"
grep -q "ramptime: 4" "$WORK_DIR/chain_output/runC/config.yaml"

mkdir -p "$WORK_DIR/sweep_nofastscan_direct"
cat > "$WORK_DIR/sweep_nofastscan_direct/config.yaml" <<EOF
# TDGL + Thermal No-FastScan Configuration

sweep:
  Jpeak: 0.2
  n_current_points: 2
  input_folder: none
  stable_time: 8.0
  dt_snapshots: 1.0
  skip_ratio: 0.1
  run_up: true
  run_down: false

outputs:
  generate_mp4: false
EOF
printf 'placeholder\n' > "$WORK_DIR/sweep_nofastscan_direct/current_Je0.1000_up.h5"

"$ROOT_DIR/package_continuation_scan_runs.sh" \
  --input_folder "$WORK_DIR/sweep_nofastscan_direct" \
  --output_folder "$WORK_DIR/direct_output" \
  --start 0.1 \
  --end 0.2 \
  --stable 40 \
  --ramptime 6

test -f "$WORK_DIR/direct_output/sweep_nofastscan_direct/config.yaml"
grep -q "input_folder: ../input_folder" "$WORK_DIR/direct_output/sweep_nofastscan_direct/config.yaml"
grep -q "stable: 40" "$WORK_DIR/direct_output/sweep_nofastscan_direct/config.yaml"
grep -q "ramptime: 6" "$WORK_DIR/direct_output/sweep_nofastscan_direct/config.yaml"

mkdir -p "$WORK_DIR/direct_nested_input/sweep_nofastscan_nested"
cat > "$WORK_DIR/direct_nested_input/config.yaml" <<EOF
# TDGL + Thermal No-FastScan Configuration

sweep:
  Jpeak: 0.2
  n_current_points: 2
  input_folder: none
  stable_time: 8.0
  dt_snapshots: 1.0
  skip_ratio: 0.1
  run_up: true
  run_down: false

outputs:
  generate_mp4: false
EOF
printf 'placeholder\n' > "$WORK_DIR/direct_nested_input/sweep_nofastscan_nested/current_Je0.1000_up.h5"

"$ROOT_DIR/package_continuation_scan_runs.sh" \
  --input_folder "$WORK_DIR/direct_nested_input" \
  --output_folder "$WORK_DIR/direct_nested_output" \
  --start 0.1 \
  --end 0.2 \
  --stable 40 \
  --ramptime 6

test -f "$WORK_DIR/direct_nested_output/direct_nested_input/config.yaml"
grep -q "input_folder: ../input_folder/sweep_nofastscan_nested" "$WORK_DIR/direct_nested_output/direct_nested_input/config.yaml"
