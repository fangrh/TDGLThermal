#!/usr/bin/env python3
"""Generate the reviewer TDGL one-factor-at-a-time configuration matrix."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    name: str
    config_path: Path
    changes: tuple[tuple[str, str, float], ...]


def _key_matches(text: str, section: str, key: str) -> list[tuple[int, re.Match[str]]]:
    current_section: str | None = None
    matches: list[tuple[int, re.Match[str]]] = []
    key_pattern = re.compile(
        rf"^(\s+){re.escape(key)}:\s*([^\s#]+)(\s*(?:#.*)?)$"
    )

    for index, line in enumerate(text.splitlines()):
        section_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(?:#.*)?$", line)
        if section_match:
            current_section = section_match.group(1)
            continue
        if line and not line[0].isspace() and not line.lstrip().startswith("#"):
            current_section = None
        if current_section == section:
            key_match = key_pattern.match(line)
            if key_match:
                matches.append((index, key_match))
    return matches


def read_yaml_scalar(text: str, section: str, key: str) -> float:
    matches = _key_matches(text, section, key)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {section}.{key} definition, found {len(matches)}"
        )
    try:
        return float(matches[0][1].group(2))
    except ValueError as exc:
        raise ValueError(f"{section}.{key} is not numeric") from exc


def replace_yaml_scalar(text: str, section: str, key: str, value: float) -> str:
    matches = _key_matches(text, section, key)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {section}.{key} definition, found {len(matches)}"
        )
    index, match = matches[0]
    lines = text.splitlines()
    indent, suffix = match.group(1), match.group(3)
    lines[index] = f"{indent}{key}: {float(value)!r}{suffix}"
    return "\n".join(lines) + "\n"


def generate_configs(
    baseline_path: Path, matrix_path: Path, output_dir: Path
) -> list[RunConfig]:
    baseline_text = baseline_path.read_text(encoding="utf-8")
    if read_yaml_scalar(baseline_text, "thermal", "k_eff") != 5.0:
        raise ValueError("baseline thermal.k_eff must be 5.0")

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    baseline_name = matrix["baseline"]
    variants = matrix["variants"]
    names = [baseline_name, *(variant["name"] for variant in variants)]
    if len(names) != len(set(names)):
        raise ValueError("run names must be unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[RunConfig] = []

    baseline_dir = output_dir / baseline_name
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_config = baseline_dir / "config.yaml"
    baseline_config.write_text(baseline_text, encoding="utf-8", newline="\n")
    runs.append(RunConfig(baseline_name, baseline_config, ()))

    for variant in variants:
        section = str(variant["section"])
        key = str(variant["key"])
        value = float(variant["value"])
        variant_text = replace_yaml_scalar(baseline_text, section, key, value)
        variant_dir = output_dir / variant["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        config_path = variant_dir / "config.yaml"
        config_path.write_text(variant_text, encoding="utf-8", newline="\n")
        runs.append(
            RunConfig(
                name=variant["name"],
                config_path=config_path,
                changes=((section, key, value),),
            )
        )

    manifest = {
        "baseline": baseline_name,
        "runs": [
            {
                "name": run.name,
                "config": run.config_path.relative_to(output_dir).as_posix(),
                "changes": [
                    {"section": section, "key": key, "value": value}
                    for section, key, value in run.changes
                ],
            }
            for run in runs
        ],
    }
    (output_dir / "generated_matrix.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = generate_configs(args.baseline, args.matrix, args.output)
    print(f"Generated {len(runs)} reviewer sensitivity configs in {args.output}")


if __name__ == "__main__":
    main()
