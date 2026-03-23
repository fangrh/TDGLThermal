#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  package_continuation_runs.sh \
    --input_folder <input-folder> \
    --output_folder <output-folder> \
    --extend_run <stable-time> \
    --skip_ratio <skip-ratio>
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

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INPUT_FOLDER=""
OUTPUT_FOLDER=""
EXTEND_RUN=""
SKIP_RATIO=""

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
        --extend_run)
            EXTEND_RUN=$2
            shift 2
            ;;
        --skip_ratio)
            SKIP_RATIO=$2
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

if [ -z "$INPUT_FOLDER" ] || [ -z "$OUTPUT_FOLDER" ] || [ -z "$EXTEND_RUN" ] || [ -z "$SKIP_RATIO" ]; then
    usage >&2
    exit 1
fi

INPUT_FOLDER=$(realpath "$INPUT_FOLDER")
mkdir -p "$OUTPUT_FOLDER"
OUTPUT_FOLDER=$(realpath "$OUTPUT_FOLDER")

for path in \
    "$ROOT_DIR/current_sweep_thermal_nofastscan.jl" \
    "$ROOT_DIR/animation_data_utils.jl" \
    "$ROOT_DIR/submit.sh"; do
    require_file "$path"
done

ln -sfn "$INPUT_FOLDER" "$OUTPUT_FOLDER/input_folder"

run_names=()

while IFS= read -r run_dir; do
    run_name=$(basename "$run_dir")
    target_dir="$OUTPUT_FOLDER/$run_name"
    mkdir -p "$target_dir"
    run_names+=("$run_name")

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

    cp "$ROOT_DIR/current_sweep_thermal_nofastscan.jl" "$target_dir/current_sweep_thermal_nofastscan.jl"
    cp "$ROOT_DIR/animation_data_utils.jl" "$target_dir/animation_data_utils.jl"
    cp "$ROOT_DIR/submit.sh" "$target_dir/submit.sh"
    cp "$source_config" "$target_dir/config.yaml"

    replace_yaml_value "$target_dir/config.yaml" "stable_time" "$EXTEND_RUN"
    replace_yaml_value "$target_dir/config.yaml" "skip_ratio" "$SKIP_RATIO"
    replace_yaml_value "$target_dir/config.yaml" "input_folder" "../input_folder/$run_name/$data_rel"

    cat > "$target_dir/run.sh" <<EOF
#!/bin/bash
set -euo pipefail
SCRIPT_DIR=\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)
cd "\$SCRIPT_DIR"
sbatch submit.sh config.yaml
EOF
    chmod +x "$target_dir/run.sh"
done < <(find "$INPUT_FOLDER" -mindepth 1 -maxdepth 1 -type d | sort)

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

echo "Packaged continuation runs in $OUTPUT_FOLDER"
