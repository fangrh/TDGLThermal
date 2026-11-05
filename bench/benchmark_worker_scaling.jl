"""Persistent-worker throughput benchmark for the CPU backends.

Run with a Julia process started using ``julia -p N --project=.``.  The
workers are kept alive for the complete scan so process startup is not part of
the per-point timings.
"""

using Distributed
const _WS_SCRIPT_START = time()
using Random, SHA, JSON3, Dates, Printf, Sockets
const _WS_SOURCE_HASH=bytes2hex(SHA.sha256(read(@__FILE__)))

function _ws_arg(name, default=nothing)
    flag = "--" * name
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && return default === nothing ? error("missing value for $flag") : default
    ARGS[i + 1]
end

function _ws_bool(name, default=false)
    value = _ws_arg(name, nothing)
    value === nothing && return any(==("--" * name), ARGS)
    lowercase(string(value)) in ("1", "true", "yes", "on")
end

# The master loads the benchmark once before workers do so.  fixtures.jl then
# sees the already-loaded TDGL definitions on each worker and does not nest the
# production include/broadcast machinery.
const _WS_CPU_FILE = joinpath(@__DIR__, "benchmark_cpu.jl")
include(_WS_CPU_FILE)
nprocs() > 1 || error("benchmark_worker_scaling.jl requires Julia started with -p N (N > 0)")
for w in workers()
    remotecall_fetch(Base.include, w, Main, _WS_CPU_FILE)
end
@everywhere using Sockets
@everywhere using YAML
# The production file's config helpers are outside its @everywhere block.
# Evaluate just that helper section on each persistent worker; including the
# whole production file here would redefine TDGL types and nested workers.
const _WS_MODEL_FILE = joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl")
const _WS_MODEL_TEXT = read(_WS_MODEL_FILE, String)
const _WS_PARAMS_END = findfirst("const DEFAULT_LOCAL_N_WORKERS", _WS_MODEL_TEXT)
_WS_PARAMS_END === nothing && error("production parameter boundary not found")
const _WS_PARAMS_SOURCE = _WS_MODEL_TEXT[1:first(_WS_PARAMS_END)-1]
for w in workers()
    remotecall_fetch(Base.include_string, w, Main, _WS_PARAMS_SOURCE, _WS_MODEL_FILE * ":params")
end
const _WS_HELPER_START = findfirst("function load_config_yaml", _WS_MODEL_TEXT)
_WS_HELPER_START === nothing && error("load_config_yaml helper not found")
const _WS_HELPER_END = findfirst("function discover_input_folder_points", _WS_MODEL_TEXT)
_WS_HELPER_END === nothing && error("config helper boundary not found")
const _WS_HELPERS_SOURCE = _WS_MODEL_TEXT[first(_WS_HELPER_START):first(_WS_HELPER_END)-1]
for w in workers()
    remotecall_fetch(Base.include_string, w, Main, _WS_HELPERS_SOURCE, _WS_MODEL_FILE * ":helpers")
end

@everywhere function _ws_worker_info(config_file, cpu_file, fixture_file, model_file, candidate_file)
    hash_file(path) = isfile(path) ? bytes2hex(SHA.sha256(read(path))) : "missing"
    (worker_id=myid(), pid=getpid(), host=Sockets.gethostname(), julia_version=string(VERSION),
     machine=Sys.MACHINE, cpu_threads=Sys.CPU_THREADS, julia_threads=Threads.nthreads(),
     total_memory=Sys.total_memory(), config_source_hash=hash_file(config_file),
     cpu_source_hash=hash_file(cpu_file), fixture_source_hash=hash_file(fixture_file),
     model_source_hash=hash_file(model_file), candidate_source_hash=hash_file(candidate_file))
end

@everywhere function _ws_warm_worker(config_path, out_dir, noise, preparation, nx, ny,
                                     seed, warm_horizon, tolerance, skip_ratio,
                                     snapshot_interval)
    mkpath(out_dir)
    c = make_case(config_path; Je=0.18, noise=noise, seed=seed,
                  preparation=preparation, Nx=nx, Ny=ny)
    for backend in ("reference", "cpu_compatible")
        warmdir = joinpath(out_dir, "worker$(myid())", backend)
        wrhs = _warm_rhs(c, backend)
        run_case(c, backend; horizon=warm_horizon, tolerance=tolerance,
                 output_dir=warmdir, save_fields=true,
                 snapshot_interval=snapshot_interval, skip_ratio=skip_ratio,
                 warm_rhs=wrhs)
    end
    true
end

@everywhere function _ws_run_job(job)
    c = make_case(job.config_path; Je=job.Je, noise=job.noise, seed=job.seed,
                  preparation=job.preparation, Nx=job.Nx, Ny=job.Ny)
    t0 = time()
    manifest = run_case(c, job.backend; horizon=job.horizon, tolerance=job.tolerance,
                        output_dir=job.output_dir, save_fields=true,
                        snapshot_interval=job.snapshot_interval, skip_ratio=job.skip_ratio)
    elapsed = time() - t0
    (; job_id=job.job_id, repetition=job.repetition, phase=job.phase,
       current_index=job.current_index, sample=job.sample, backend=job.backend,
       Je=job.Je, seed=job.seed, worker_id=myid(),
       point_wall_seconds=elapsed, wall_seconds=manifest.wall_seconds,
       rhs_seconds=manifest.rhs_seconds, rhs_calls=manifest.rhs_calls,
       accepted_steps=manifest.accepted_steps, rejected_steps=manifest.rejected_steps,
       complete=manifest.complete, status=manifest.status, output_path=manifest.output_path,
       initial_hash=manifest.initial_hash,initial_vector_hash=manifest.initial_vector_hash,
       model_hash=manifest.model_hash,rhs_source_hash=manifest.rhs_source_hash,
       config_hash=manifest.config_hash,output_hash=manifest.output_hash)
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = _ws_arg("out"); out === nothing && error("--out is required")
    config_path = abspath(_ws_arg("config", joinpath(@__DIR__, "..", "config.yaml")))
    horizon = parse(Float64, _ws_arg("horizon", "100"))
    repetitions = parse(Int, _ws_arg("repetitions", "3")); repetitions > 0 || error("repetitions must be positive")
    samples_per_current = parse(Int, _ws_arg("samples-per-current", "3")); samples_per_current > 0 || error("samples-per-current must be positive")
    noise = parse(Float64, _ws_arg("noise", "0.001"))
    preparation = Symbol(_ws_arg("preparation", "configured_down"))
    tolerance = parse(Float64, _ws_arg("tolerance", "1e-4"))
    skip_ratio = parse(Float64, _ws_arg("skip", "0.9"))
    snapshot_interval = parse(Float64, _ws_arg("snapshot-interval", "1"))
    nx = parse(Int, _ws_arg("nx", "60")); ny = parse(Int, _ws_arg("ny", "30"))
    base_seed = parse(Int, _ws_arg("seed", "2026110401"))
    full_scan = _ws_bool("full-scan", false)
    horizon > 0 || error("horizon must be positive")
    mkpath(out)

    # The pilot is deliberately index-based: it is 12 of the 99-point Je grid.
    currents = collect(range(0.0, 0.3; length=99))
    pilot_indices = [1, 10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 99]
    indices = full_scan ? collect(eachindex(currents)) : pilot_indices
    cpu_file = _WS_CPU_FILE
    fixture_file = joinpath(@__DIR__, "fixtures.jl")
    model_file = joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl")
    candidate_file = joinpath(@__DIR__, "..", "src", "cpu_compatible.jl")
    infos = [remotecall_fetch(_ws_worker_info, w, config_path, cpu_file, fixture_file,
                              model_file, candidate_file) for w in workers()]

    warmup_horizon = min(horizon, 2.0)
    cold_setup_seconds = time() - _WS_SCRIPT_START
    # One warmup per persistent worker, with both backends, before any timed phase.
    warmups = [remotecall(_ws_warm_worker, w, config_path, joinpath(out, "warmup"), noise,
                         preparation, nx, ny, base_seed, warmup_horizon, tolerance,
                         skip_ratio, snapshot_interval) for w in workers()]
    fetch.(warmups)

    phases = NamedTuple[]
    all_rows = NamedTuple[]
    for repetition in 1:repetitions
        order = isodd(repetition) ? ("reference", "cpu_compatible") :
            ("cpu_compatible", "reference")
        for (phase_number, backend) in enumerate(order)
            jobs = NamedTuple[]
            job_id = 0
            for index in indices
                for sample in 1:samples_per_current
                    job_id += 1
                    # The same seed is used by both backend phases, while each
                    # sample is an explicitly isolated make_case/run_case job.
                    seed = base_seed + 100 * (repetition-1) + (sample-1)
                    push!(jobs, (job_id=job_id, repetition=repetition, phase=phase_number,
                        current_index=index, sample=sample, backend=backend,
                        Je=currents[index], noise=noise, seed=seed,
                        preparation=preparation, config_path=config_path, Nx=nx, Ny=ny,
                        horizon=horizon, tolerance=tolerance, skip_ratio=skip_ratio,
                        snapshot_interval=snapshot_interval,
                        output_dir=joinpath(out, "rep$(repetition)", "phase$(phase_number)",
                                            "job$(job_id)_idx$(index)_sample$(sample)_Je$(currents[index])")))
                end
            end
            timed = @timed rows = pmap(_ws_run_job, jobs)
            append!(all_rows, rows)
            push!(phases, (repetition=repetition, phase=phase_number, backend=backend,
                jobs=length(jobs), jobs_per_phase=length(jobs),
                seeds=unique([j.seed for j in jobs]), total_phase_wall_seconds=timed.time,
                point_wall_seconds=[r.point_wall_seconds for r in rows],
                rhs_calls=[r.rhs_calls for r in rows], rows=rows))
        end
    end

    result = (schema_version="tdgl-worker-scaling-v1", backend_order="alternating",
        full_scan=full_scan, current_grid=currents, selected_indices=indices,
        samples_per_current=samples_per_current, horizon=horizon, repetitions=repetitions, noise=noise,
        preparation=string(preparation), tolerance=tolerance,
        skip_ratio=skip_ratio, snapshot_interval=snapshot_interval,
        actual_nworkers=nworkers(), worker_ids=workers(), worker_info=infos,
        cold_setup_seconds=cold_setup_seconds, script_start_seconds=_WS_SCRIPT_START,
        all_seeds=sort(unique([r.seed for r in all_rows])),
        julia_version=string(VERSION), config_path=config_path,
        config_source_hash=isfile(config_path) ? bytes2hex(SHA.sha256(read(config_path))) : "missing",
        benchmark_source_hash=_WS_SOURCE_HASH,seed_base=base_seed,
        phases=phases, rows=all_rows, created_utc=string(now(UTC)))
    open(joinpath(out, "worker_scaling.json"), "w") do io
        write(io, JSON3.write(result)); write(io, '\n')
    end
    @info "worker scaling complete" out actual_nworkers=nworkers() jobs=length(all_rows)
end
