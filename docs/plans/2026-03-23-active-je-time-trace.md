# Active Je Time Trace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change the animation voltage-vs-time panel so each rendered frame shows only the active frame's `Je` trace.

**Architecture:** Keep the existing stored animation data format unchanged. Restrict the plotting branch in `generate_single_Je_animation_remote` so panel `p4` renders only the selected frame's time trace for the active direction and `Je`, then verify that selection logic with a focused regression test.

**Tech Stack:** Julia, `Test`, `Plots`

---

### Task 1: Add regression coverage for active-Je trace selection

**Files:**
- Create: `test/animation_time_trace_selection.jl`
- Modify: `current_sweep_thermal_nofastscan.jl`

**Step 1: Write the failing test**

```julia
using Test

@testset "active Je time trace selection" begin
    v_t_up = Dict(
        0.1 => [(1.0, 10.0)],
        0.2 => [(2.0, 20.0)],
    )

    selected = select_animation_time_trace(v_t_up, "up", 0.2)

    @test selected == [(2.0, 20.0)]
end
```

**Step 2: Run test to verify it fails**

Run: `julia test/animation_time_trace_selection.jl`
Expected: `FAIL` because `select_animation_time_trace` does not exist yet.

**Step 3: Write minimal implementation**

```julia
function select_animation_time_trace(v_t_by_direction, direction, Je)
    dir_v_t_data = direction == "up" ? v_t_by_direction[1] : v_t_by_direction[2]
    return get(dir_v_t_data, Je, Tuple{Float64, Float64}[])
end
```

**Step 4: Run test to verify it passes**

Run: `julia test/animation_time_trace_selection.jl`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test/animation_time_trace_selection.jl current_sweep_thermal_nofastscan.jl docs/plans/2026-03-23-active-je-time-trace.md
git commit -m "fix: limit animation time trace to active Je"
```

### Task 2: Use the selected trace in the animation panel

**Files:**
- Modify: `current_sweep_thermal_nofastscan.jl`

**Step 1: Update the plotting branch**

Replace the `for (Je_val, V_t_array) in sort(...)` loop in the `p4` panel with a single lookup for the active `Je`.

**Step 2: Run the regression test**

Run: `julia test/animation_time_trace_selection.jl`
Expected: `PASS`

**Step 3: Run a syntax-level verification**

Run: `julia --project=. -e 'include("current_sweep_thermal_nofastscan.jl")'`
Expected: exit code `0`

**Step 4: Commit**

```bash
git add test/animation_time_trace_selection.jl current_sweep_thermal_nofastscan.jl docs/plans/2026-03-23-active-je-time-trace.md
git commit -m "fix: limit animation time trace to active Je"
```
