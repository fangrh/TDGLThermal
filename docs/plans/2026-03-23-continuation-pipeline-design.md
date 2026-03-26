# Continuation Pipeline Design

**Date:** 2026-03-23

## Goal

Add a continuation workflow that resumes from an existing sweep output folder, selects the saved current at or below a requested `--start`, and continues through existing saved current values up to `--end`.

The new workflow must preserve the same output structure and `.h5` layout as the current no-fastscan sweep outputs so the generated continuation output can be used as input for another continuation run without conversion.

## Constraints

- The current list must come from the existing `.h5` files in the provided input folder.
- Start selection is the largest saved current value less than or equal to `--start`.
- Output includes only the resumed range, not the full original sweep.
- The stable run for each current must branch from the post-ramp state.
- A dedicated ramp process continues forward without waiting for stable runs to finish.
- `config.yaml` in the packaged output stays compatible with the current repository workflow.

## Options Considered

### Option 1: Dedicated continuation runner and dedicated packager

Create a new Julia entrypoint for continuation execution and a new shell packager for preparing and submitting those runs.

Pros:
- Keeps continuation logic isolated from the existing sweep path.
- Minimizes regression risk in the current no-fastscan runner.
- Makes the pipeline execution model explicit.

Cons:
- Adds one more Julia script and one more shell entrypoint.

### Option 2: Add a continuation mode to the existing no-fastscan runner

Extend the current sweep script with mode flags and continuation branching logic.

Pros:
- Fewer entrypoint files.

Cons:
- Mixes two different execution models in one large file.
- Increases risk of breaking the existing sweep path.
- Harder to test and reason about.

### Option 3: Drive continuation mostly from shell

Generate per-step continuation tasks in shell while keeping Julia logic minimal.

Pros:
- Small shell wrapper changes at first glance.

Cons:
- Pushes simulation-state orchestration into packaging glue.
- Poor fit for branch-and-continue solver logic.

## Recommended Approach

Use Option 1.

Add a dedicated continuation Julia script and a dedicated submission/packaging shell script, while reusing the existing data format and helper code. This keeps the continuation implementation local to the new workflow and avoids entangling the current sweep path with pipeline-specific control flow.

## Execution Model

1. Read the input folder and discover available `current_Je*.h5` files.
2. Extract the saved current values and sort them in sweep order.
3. Choose the largest saved current `<= --start`.
4. Build the continuation target list from that chosen current through the largest saved current `<= --end`.
5. Load the chosen current's saved state as the initial pipeline state.
6. Maintain one ramp pipeline that advances from one target current to the next using `--ramptime`.
7. After each ramp completes, dispatch a separate stable worker from the post-ramp state for `--stable` time at that current.
8. The ramp pipeline immediately continues from the same post-ramp state to the next target current.
9. Each stable worker writes the normal per-current `.h5` output in the same structure used today.

The stable runs are therefore branches from the ramp trajectory. The next ramp step uses the post-ramp state, not the end state of the stable worker.

## Data Contract

- The packaged output directory keeps the same top-level layout used by the current workflow.
- The continuation output folder contains the same style of `config.yaml`, support scripts, and per-current `.h5` files.
- The `.h5` naming scheme and internal datasets remain unchanged.
- Existing downstream tooling should not need to detect whether a run was original or continued.
- A continuation output folder must be valid as the input folder for another continuation package/run.

## CLI Contract

### Julia runner

Expected inputs:

- `config.yaml`
- `--input_folder`
- `--start`
- `--end`
- `--stable`
- `--ramptime`

The runner should validate that:

- the input folder exists
- at least one valid `.h5` file is present
- a saved current exists at or below `--start`
- `--end` reaches at least one saved current in the discovered sequence

### Packaging script

Expected responsibilities:

- copy the continuation Julia runner and shared helper files
- preserve the familiar packaged directory layout
- rewrite `config.yaml` only for continuation-specific values and paths
- generate a local `run.sh`
- submit each packaged run with `sbatch`

## Testing Strategy

### Shell regression coverage

Add a new shell test for the continuation packager to verify:

- expected files are copied into each packaged run
- `config.yaml` is rewritten with continuation inputs
- launcher scripts are created
- `sbatch` is invoked once per packaged run

### Julia-side behavior coverage

Add focused coverage for:

- selecting the largest saved current `<= --start`
- trimming the continuation sequence at `--end`
- preserving the discovered saved-current order

Where practical, isolate current-discovery and selection logic into testable helper functions.

## Risks

- The existing sweep script is large, so shared code extraction should stay minimal and targeted.
- Pipeline execution can introduce race or memory-pressure issues if stable workers are launched faster than available resources.
- The output layout must stay exact, otherwise the continuation chain will break later.

## Success Criteria

- A user can point the new packager at an existing sweep folder, provide `--start`, `--end`, `--stable`, and `--ramptime`, and obtain a submitted continuation run.
- The continuation runner uses the saved current at or below `--start`.
- Stable branches run from post-ramp states while the ramp pipeline continues forward.
- The produced output structure can be used directly for another continuation run.
