# README And Gitignore Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add repository documentation and git ignore rules so source files are tracked while generated data and run artifacts stay out of version control.

**Architecture:** Create a root `README.md` describing the project structure, run flow, and continuation packaging workflow. Add a root `.gitignore` that excludes generated data folders, sweep outputs, HDF5 data, SLURM logs, and temporary artifacts while keeping source, docs, and tests tracked.

**Tech Stack:** Markdown, Git ignore patterns, Bash/Julia workflow references

---

### Task 1: Add repository README

**Files:**
- Create: `README.md`

**Step 1: Write the content**

Document:
- project purpose
- key scripts (`current_sweep_thermal_nofastscan.jl`, `package_and_submit.sh`, `package_continuation_runs.sh`)
- basic run command
- continuation packaging example
- note that generated data is intentionally ignored by git

**Step 2: Verify the README exists and is readable**

Run: `sed -n '1,220p' README.md`
Expected: expected sections present and command exits `0`

**Step 3: Commit**

```bash
git add README.md docs/plans/2026-03-23-readme-and-gitignore.md
git commit -m "docs: add repository readme"
```

### Task 2: Add git ignore rules for generated data

**Files:**
- Create: `.gitignore`

**Step 1: Write ignore rules**

Ignore:
- `changeT*/`
- `sweep_nofastscan_*/`
- `frames/`
- `*.h5`
- `*_slurm.out`
- `*_slurm.err`
- common temp/editor artifacts

Keep tracked:
- source `.jl` and `.sh` files
- `docs/`
- `test/`

**Step 2: Verify ignore file content**

Run: `sed -n '1,220p' .gitignore`
Expected: expected patterns present and command exits `0`

**Step 3: Commit**

```bash
git add .gitignore README.md docs/plans/2026-03-23-readme-and-gitignore.md
git commit -m "chore: ignore generated run data"
```
