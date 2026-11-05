# Matched reference/CPU benchmark runner.
using Random, SHA, JSON3, Statistics, Dates, Printf, Serialization
using HDF5
include(joinpath(@__DIR__, "fixtures.jl"))
# Harness-only checked stepping.  This does not alter the frozen model/RHS.
include(joinpath(@__DIR__, "streaming_iteration_guard.jl"))

_backend_fn(name) = isdefined(Main, :resolve_rhs_backend) ? resolve_rhs_backend(name) :
    (name == "reference" ? fdm_rhs_thermal! : error("cpu_compatible is unavailable"))

function _write_json(path, x)
    open(path, "w") do io; write(io, JSON3.write(x)); write(io, '\n'); end
end

function _write_row(path, x)
    keys = (:backend, :initial_hash, :Je, :noise, :horizon, :tolerance, :save_fields,
            :status, :complete, :final_time, :wall_seconds, :rhs_seconds,
            :rhs_allocated, :accepted_steps, :rejected_steps, :rhs_calls, :output_schema)
    open(path, "w") do io
        println(io, join(string.(keys), ','))
        vals = [replace(string(get(x, k, nothing)), ',' => ';') for k in keys]
        println(io, join(vals, ','))
    end
end

function _validate_output(path, horizon, p)
    isfile(path) || return false
    h5open(path, "r") do f
        required = ("times", "V", "psi_avg", "T_avg", "psi_real", "psi_imag", "T", "Ax", "Ay")
        all(haskey(f, k) for k in required) || return false
        ts = read(f["times"]); n = length(ts)
        isempty(ts) && return false
        all(isfinite, ts) && isapprox(ts[end], horizon; atol=8eps(max(horizon, 1.0))) || return false
        expected = Dict("V"=>(n,), "psi_avg"=>(n,), "T_avg"=>(n,),
            "psi_real"=>(p.Nx+1,p.Ny+1,n), "psi_imag"=>(p.Nx+1,p.Ny+1,n),
            "T"=>(p.Nx+1,p.Ny+1,n), "Ax"=>(p.Nx,p.Ny+1,n), "Ay"=>(p.Nx+1,p.Ny,n))
        for k in keys(expected)
            size(f[k]) == expected[k] || return false
            all(isfinite, read(f[k])) || return false
        end
        true
    end
end

function _stats(sol)
    s = hasproperty(sol, :stats) ? sol.stats : sol.destats
    (accepted_steps=getproperty(s, :naccept), rejected_steps=getproperty(s, :nreject),
     rhs_calls=getproperty(s, :nf))
end

function _run_stream(case, backend, horizon, tolerance, path, snapshot_interval, skip_ratio;
                      rng_state=nothing)
    p = with_current(case.params, case.Je)
    # Rebuild from the immutable packed bytes so reference and candidate never
    # share a state object (the streaming solver applies boundary conditions in place).
    state = initialize_state_clean(p)
    vector_to_state!(state, copy(case.u0), p)
    rng_state === nothing ? Random.seed!(case.seed) : copy!(Random.default_rng(), rng_state)
    fn = run_stable_simulation_streaming
    return fn(state, p, horizon, snapshot_interval, skip_ratio, path, case.Je, "benchmark";
              backend=backend, abstol=tolerance, reltol=tolerance)
end

function _run_memory(case, backend, horizon, tolerance)
    case.noise == 0.0 || error("memory-only mode is deterministic-only; use save_fields=true for noise")
    p = with_current(case.params, case.Je)
    Random.seed!(case.seed)
    u0 = copy(case.u0); cache = TDGLThermalCache(p)
    rhs = _backend_fn(backend); du = similar(u0)
    Random.seed!(case.seed); rhs(du, u0, cache, 0.0)
    Random.seed!(case.seed)
    rhs_time = @elapsed rhs(du, u0, cache, 0.0)
    rhs_alloc = @allocated rhs(du, u0, cache, 0.0)
    times = collect(0.0:1.0:horizon)
    times[end] == horizon || push!(times, horizon)
    Random.seed!(case.seed)
    sol_time = @elapsed sol = solve(ODEProblem(rhs, u0, (0.0, horizon), cache), Tsit5();
                                    abstol=tolerance, reltol=tolerance, saveat=times,
                                    save_everystep=false)
    status = string(sol.retcode)
    finite = all(isfinite, sol.u[end])
    (status=status, complete=finite && isapprox(sol.t[end], horizon; atol=8eps(horizon)),
     final_time=sol.t[end], accepted_steps=_stats(sol).accepted_steps,
     rejected_steps=_stats(sol).rejected_steps, rhs_calls=_stats(sol).rhs_calls,
     solver_seconds=sol_time, rhs_seconds=rhs_time, rhs_allocated=rhs_alloc,
     output_schema="trajectory_without_fields")
end

function _warm_rhs(case, backend)
    p = with_current(case.params, case.Je); rhs = _backend_fn(backend)
    u = copy(case.u0); du = similar(u); cache = TDGLThermalCache(p)
    Random.seed!(case.seed); rhs(du, u, cache, 0.0)
    Random.seed!(case.seed)
    elapsed = @elapsed rhs(du, u, cache, 0.0)
    alloc = @allocated rhs(du, u, cache, 0.0)
    (rhs_seconds=elapsed, rhs_allocated=alloc)
end

function run_case(case, backend::String; horizon::Float64, tolerance::Float64,
                  output_dir::String, save_fields::Bool=true,
                  snapshot_interval::Float64=1.0, skip_ratio::Float64=0.9,
                  warm_rhs=nothing, rng_state=nothing)
    backend in ("reference", "cpu_compatible") || error("unsupported backend: $backend")
    horizon > 0 && tolerance > 0 || error("horizon and tolerance must be positive")
    snapshot_interval > 0 || error("snapshot interval must be positive")
    0 <= skip_ratio <= 1 || error("skip ratio must be in [0,1]")
    mkpath(output_dir)
    out = joinpath(output_dir, "$(backend)_Je$(case.Je)_seed$(case.seed).h5")
    packed_path = joinpath(output_dir, "$(backend)_Je$(case.Je)_seed$(case.seed)_initial.bin")
    open(packed_path, "w") do io; write(io, reinterpret(UInt8, case.u0)); end
    packed_hash = bytes2hex(SHA.sha256(read(packed_path)))
    t0 = time(); result = save_fields ? _run_stream(case, backend, horizon, tolerance, out, snapshot_interval, skip_ratio;
                                                     rng_state=rng_state) :
        _run_memory(case, backend, horizon, tolerance)
    wall = time() - t0
    vals = result isa NamedTuple ? result : (; result...)
    haskey(vals, :complete) && haskey(vals, :status) && haskey(vals, :final_time) ||
        error("streaming result lacks complete/status/final_time statistics")
    complete = vals.complete
    status = string(vals.status)
    isfinite(wall) && wall > 0 || error("non-positive benchmark wall time")
    isfinite(vals.final_time) && isapprox(vals.final_time, horizon; atol=8eps(max(horizon, 1.0))) ||
        error("solver did not reach requested horizon")
    complete || error("solver reported incomplete trajectory")
    complete && (save_fields ? _validate_output(out, horizon, case.params) : !isfile(out)) || error("incomplete output for $backend")
    manifest = (schema_version="tdgl-cpu-benchmark-v1", backend=backend,
        role=backend == "reference" ? "corrected_reference" : "compatible_candidate",
        config_path=case.config_path, config_hash=case.config_hash, model_hash=case.model_hash,
        initial_hash=case.initial_hash,
        initial_state_source=case.initial_state_source,
        initial_source_hash=case.initial_source_hash,
        seed=case.seed, Je=case.Je, noise=case.noise, preparation=string(case.preparation),
        Nx=case.params.Nx, Ny=case.params.Ny, hx=case.params.hx, hy=case.params.hy,
        horizon=horizon, tolerance=tolerance, snapshot_interval=snapshot_interval, skip_ratio=skip_ratio,
        solver="Tsit5", precision="ComplexF64/Float64",
        save_fields=save_fields, status=status, complete=complete, final_time=get(vals, :final_time, horizon),
        wall_seconds=wall, rhs_seconds=warm_rhs === nothing ? get(vals, :rhs_seconds, nothing) : warm_rhs.rhs_seconds,
        rhs_allocated=get(warm_rhs === nothing ? vals : warm_rhs, :rhs_allocated, nothing),
        accepted_steps=get(vals, :accepted_steps, nothing),
        rejected_steps=get(vals, :rejected_steps, nothing), rhs_calls=get(vals, :rhs_calls, nothing),
        output_schema=get(vals, :output_schema, save_fields ? "streamed_hdf5_fields" : "trajectory_without_fields"),
        output_path=save_fields ? out : nothing, rhs_warm_seconds=warm_rhs === nothing ? nothing : warm_rhs.rhs_seconds,
        output_hash=save_fields ? bytes2hex(SHA.sha256(read(out))) : nothing,
        stream_maxiters_override=stream_maxiters_override(),
        stream_guard_source_hash=STREAM_GUARD_SOURCE_HASH,
        initial_vector_path=packed_path, initial_vector_hash=packed_hash,
        frozen_reference_commit="6ce36684a46c8e9938b75cd843c26148fe71e008",
        environment_hash=bytes2hex(SHA.sha256(string(VERSION, ":", Sys.MACHINE))),
        created_utc=string(now(UTC)), julia_version=case.julia_version, machine=case.machine,
        rhs_source_hash=case.rhs_source_hash,
        accuracy_status=save_fields ? "pending_fieldwise_comparison" : "deterministic_only")
    _write_json(joinpath(output_dir, "$(backend)_Je$(case.Je)_seed$(case.seed).json"), manifest)
    _write_row(joinpath(output_dir, "$(backend)_Je$(case.Je)_seed$(case.seed).csv"), manifest)
    return manifest
end

function _arg(name, default=nothing)
    i = findfirst(==("--" * name), ARGS)
    i === nothing ? default : (i == length(ARGS) ? error("missing value for --$name") : ARGS[i+1])
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = _arg("out"); out === nothing && error("--out is required")
    cfg = _arg("config", joinpath(@__DIR__, "..", "config.yaml"))
    horizon = parse(Float64, _arg("horizon", "2")); noise = parse(Float64, _arg("noise", "0.0"))
    currents = parse.(Float64, split(_arg("currents", "0.18"), ','))
    repetitions = parse(Int, _arg("repetitions", "1")); repetitions > 0 || error("repetitions must be positive")
    grid = split(_arg("grid", "60x30"), 'x'); length(grid) == 2 || error("grid must be Nx x Ny")
    tolerance = parse(Float64, _arg("tolerance", "1e-4"))
    snapshot_interval = parse(Float64, _arg("snapshot-interval", "1.0"))
    skip_ratio = parse(Float64, _arg("skip", "0.9"))
    backends = split(_arg("backends", "reference,cpu_compatible"), ',')
    all(b -> b in ("reference", "cpu_compatible"), backends) || error("unknown backend")
    length(unique(backends)) == length(backends) || error("duplicate backend")
    warmup_horizon = min(horizon, parse(Float64, _arg("warmup-horizon", "2.0")))
    warmup_horizon > 0 || error("warmup horizon must be positive")
    for Je in currents
        c = make_case(cfg; Je=Je, noise=noise, seed=parse(Int, _arg("seed", "20261104")),
                      preparation=Symbol(_arg("preparation", "cool")), Nx=parse(Int, grid[1]), Ny=parse(Int, grid[2]),
                      initial_state_path=_arg("initial-state"))
        # Compile the complete solver/streaming/HDF5 path once per backend. These
        # artifacts live outside timed repetitions and are never compared.
        for backend in backends
            warmdir = joinpath(out, "Je$(Je)", "warmup", backend)
            wrhs = _warm_rhs(c, backend)
            run_case(c, String(backend); horizon=warmup_horizon, tolerance=tolerance, output_dir=warmdir,
                     save_fields=true, snapshot_interval=snapshot_interval, skip_ratio=skip_ratio, warm_rhs=wrhs)
        end
        for rep in 1:repetitions
            order = isodd(rep) ? backends : reverse(backends)
            repout = joinpath(out, "Je$(Je)", "rep$(rep)")
            for backend in order
                wrhs = _warm_rhs(c, backend) # RHS sampling is outside timed solve
                run_case(c, String(backend); horizon=horizon, tolerance=tolerance,
                         output_dir=repout, save_fields=true, snapshot_interval=snapshot_interval,
                         skip_ratio=skip_ratio, warm_rhs=wrhs)
            end
        end
    end
end
