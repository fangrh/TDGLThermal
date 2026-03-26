#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  package_continuation_scan_runs.sh \
    --input_folder <input-folder> \
    --output_folder <output-folder> \
    --start <start-current> \
    --end <end-current> \
    --stable <stable-time> \
    --ramptime <ramp-time>
EOF
}

require_file() {
    local path=$1
    if [ ! -f "$path" ]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
}

replace_yaml_value() {
    local file=$1
    local key=$2
    local value=$3
    python3 - "$file" "$key" "$value" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
text = path.read_text()
pattern = re.compile(rf"(^\s*{re.escape(key)}:\s*).*$", re.MULTILINE)
new_text, count = pattern.subn(lambda m: f"{m.group(1)}{value}", text, count=1)
if count != 1:
    raise SystemExit(f"Could not find YAML key '{key}' in {path}")
path.write_text(new_text)
PY
}

upsert_continuation_block() {
    local file=$1
    local start_value=$2
    local end_value=$3
    local stable_value=$4
    local ramptime_value=$5
    python3 - "$file" "$start_value" "$end_value" "$stable_value" "$ramptime_value" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
start_value, end_value, stable_value, ramptime_value = sys.argv[2:6]
text = path.read_text()
text = re.sub(r"\ncontinuation_scan:\n(?:  .*\n?)*", "\n", text, flags=re.MULTILINE)
text = text.rstrip() + (
    "\n\ncontinuation_scan:\n"
    f"  start: {start_value}\n"
    f"  end: {end_value}\n"
    f"  stable: {stable_value}\n"
    f"  ramptime: {ramptime_value}\n"
)
path.write_text(text)
PY
}

find_data_dir() {
    local run_dir=$1
    local data_dir
    data_dir=$(find "$run_dir" -mindepth 1 -maxdepth 2 -type f -name 'current_Je*.h5' -printf '%h\n' | sort -u | head -n 1 || true)
    if [ -z "$data_dir" ]; then
        echo ""
        return
    fi
    realpath --relative-to="$run_dir" "$data_dir"
}

list_input_runs() {
    local input_root=$1
    local direct_data_rel
    direct_data_rel=$(find_data_dir "$input_root")

    if [ -f "$input_root/config.yaml" ] && [ -n "$direct_data_rel" ]; then
        printf '%s\n' "$input_root"
        return
    fi

    find "$input_root" -mindepth 1 -maxdepth 1 -type d | sort
}

build_packaged_input_path() {
    local input_root=$1
    local run_dir=$2
    local run_name=$3
    local data_rel=$4

    if [ "$run_dir" = "$input_root" ]; then
        if [ "$data_rel" = "." ]; then
            printf '%s\n' "../input_folder"
        else
            printf '%s\n' "../input_folder/$data_rel"
        fi
        return
    fi

    if [ "$data_rel" = "." ]; then
        printf '%s\n' "../input_folder/$run_name"
    else
        printf '%s\n' "../input_folder/$run_name/$data_rel"
    fi
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INPUT_FOLDER=""
OUTPUT_FOLDER=""
START_CURRENT=""
END_CURRENT=""
STABLE_TIME=""
RAMP_TIME=""

while [ $# -gt 0 ]; do
    case "$1" in
        --input_folder)
            INPUT_FOLDER=$2
            shift 2
            ;;
        --output_folder)
            OUTPUT_FOLDER=$2
            shift 2
            ;;
        --start)
            START_CURRENT=$2
            shift 2
            ;;
        --end)
            END_CURRENT=$2
            shift 2
            ;;
        --stable)
            STABLE_TIME=$2
            shift 2
            ;;
        --ramptime)
            RAMP_TIME=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "$INPUT_FOLDER" ] || [ -z "$OUTPUT_FOLDER" ] || [ -z "$START_CURRENT" ] || [ -z "$END_CURRENT" ] || [ -z "$STABLE_TIME" ] || [ -z "$RAMP_TIME" ]; then
    usage >&2
    exit 1
fi

INPUT_FOLDER=$(realpath "$INPUT_FOLDER")
mkdir -p "$OUTPUT_FOLDER"
OUTPUT_FOLDER=$(realpath "$OUTPUT_FOLDER")

for path in \
    "$ROOT_DIR/current_sweep_thermal_continue.jl" \
    "$ROOT_DIR/current_sweep_thermal_nofastscan.jl" \
    "$ROOT_DIR/animation_data_utils.jl" \
    "$ROOT_DIR/submit_scan.sh"; do
    require_file "$path"
done

ln -sfn "$INPUT_FOLDER" "$OUTPUT_FOLDER/input_folder"

run_names=()

while IFS= read -r run_dir; do
    run_name=$(basename "$run_dir")
    target_dir="$OUTPUT_FOLDER/$run_name"

    source_config="$run_dir/config.yaml"
    if [ ! -f "$source_config" ]; then
        echo "Skipping $run_name: missing config.yaml" >&2
        continue
    fi

    data_rel=$(find_data_dir "$run_dir")
    if [ -z "$data_rel" ]; then
        echo "Skipping $run_name: no continuation H5 files found" >&2
        continue
    fi

    mkdir -p "$target_dir"
    run_names+=("$run_name")

    cp "$ROOT_DIR/current_sweep_thermal_continue.jl" "$target_dir/current_sweep_thermal_continue.jl"
    cp "$ROOT_DIR/current_sweep_thermal_nofastscan.jl" "$target_dir/current_sweep_thermal_nofastscan.jl"
    cp "$ROOT_DIR/animation_data_utils.jl" "$target_dir/animation_data_utils.jl"
    cp "$ROOT_DIR/submit_scan.sh" "$target_dir/submit_scan.sh"
    cp "$source_config" "$target_dir/config.yaml"

    replace_yaml_value "$target_dir/config.yaml" "stable_time" "$STABLE_TIME"
    packaged_input_path=$(build_packaged_input_path "$INPUT_FOLDER" "$run_dir" "$run_name" "$data_rel")
    replace_yaml_value "$target_dir/config.yaml" "input_folder" "$packaged_input_path"
    upsert_continuation_block "$target_dir/config.yaml" "$START_CURRENT" "$END_CURRENT" "$STABLE_TIME" "$RAMP_TIME"

    cat > "$target_dir/run.sh" <<EOF
#!/bin/bash
set -euo pipefail
SCRIPT_DIR=\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)
cd "\$SCRIPT_DIR"
sbatch submit_scan.sh config.yaml
EOF
    chmod +x "$target_dir/run.sh"

    (
        cd "$target_dir"
        sbatch submit_scan.sh config.yaml
    )
done < <(list_input_runs "$INPUT_FOLDER")

cat > "$OUTPUT_FOLDER/run_all.sh" <<EOF
#!/bin/bash
set -euo pipefail
ROOT_DIR=\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)
EOF

for run_name in "${run_names[@]}"; do
    cat >> "$OUTPUT_FOLDER/run_all.sh" <<EOF
bash "\$ROOT_DIR/$run_name/run.sh"
EOF
done

chmod +x "$OUTPUT_FOLDER/run_all.sh"

echo "Packaged continuation scan runs in $OUTPUT_FOLDER"
