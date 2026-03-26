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

"$ROOT_DIR/package_continuation_runs.sh" \
  --input_folder "$ROOT_DIR/test/fixtures/package_input/changeT1" \
  --output_folder "$WORK_DIR/changeT2" \
  --extend_run 5000 \
  --skip_ratio 0.6

test -d "$WORK_DIR/changeT2/runA"
test -d "$WORK_DIR/changeT2/runB"
test -L "$WORK_DIR/changeT2/input_folder"
test -f "$WORK_DIR/changeT2/runA/config.yaml"
test -f "$WORK_DIR/changeT2/runA/current_sweep_thermal_nofastscan.jl"
test -f "$WORK_DIR/changeT2/runA/animation_data_utils.jl"
test -f "$WORK_DIR/changeT2/runA/submit.sh"
test -x "$WORK_DIR/changeT2/runA/run.sh"
test -x "$WORK_DIR/changeT2/runB/run.sh"
test -x "$WORK_DIR/changeT2/run_all.sh"
grep -q "stable_time: 5000" "$WORK_DIR/changeT2/runA/config.yaml"
grep -q "skip_ratio: 0.6" "$WORK_DIR/changeT2/runA/config.yaml"
grep -q "input_folder: ../input_folder/runA/source_data" "$WORK_DIR/changeT2/runA/config.yaml"
grep -q 'bash "$ROOT_DIR/runA/run.sh"' "$WORK_DIR/changeT2/run_all.sh"
grep -q 'bash "$ROOT_DIR/runB/run.sh"' "$WORK_DIR/changeT2/run_all.sh"
test -f "$WORK_DIR/sbatch.log"
test "$(wc -l < "$WORK_DIR/sbatch.log")" -eq 2
grep -q "^submit.sh config.yaml$" "$WORK_DIR/sbatch.log"
