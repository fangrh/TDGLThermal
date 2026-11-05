"""Compare noisy and noise-free adaptive Tsit5 step counts on archived states.

This is a deliberately simplified diagnostic: it omits streaming/output
diagnostics and does not claim noise-on/noise-off physical equivalence.
"""

using JSON3, Statistics, Random, SHA, Dates

const REPO = abspath(begin
    i = findfirst(==("--repo"), ARGS)
    i === nothing ? joinpath(@__DIR__, "..") : ARGS[i + 1]
end)

function arg_value(name; default=nothing, required=false)
    flag = "--" * name
    i = findfirst(==(flag), ARGS)
    if i === nothing
        required && error("missing required $flag")
        return default
    end
    i == length(ARGS) && error("missing value for $flag")
    return ARGS[i + 1]
end

const INPUTS_PATH = arg_value("inputs"; required=true)
const OUTPUT_PATH = abspath(arg_value("out"; required=true))
const HORIZON = parse(Float64, arg_value("horizon"; default="100"))

HORIZON > 0 || error("--horizon must be positive")
isfile(INPUTS_PATH) || error("inputs JSON not found: $INPUTS_PATH")
isfile(OUTPUT_PATH) && error("refusing to overwrite $OUTPUT_PATH")

ENV["NOFASTSCAN_SKIP_MAIN"] = "1"
ENV["NOFASTSCAN_SKIP_WORKERS"] = "1"
include(joinpath(REPO, "bench", "fixtures.jl"))

const CONFIG_PATH = joinpath(REPO, "studies", "full_curl_ndr", "config_T0.18.yaml")
const MODEL_PATH = joinpath(REPO, "current_sweep_thermal_nofastscan.jl")
const RHS_PATH = joinpath(REPO, "src", "cpu_compatible.jl")
isfile(CONFIG_PATH) || error("study config not found: $CONFIG_PATH")

sha(path) = bytes2hex(SHA.sha256(read(path)))

function solve_stats(sol)
    stats = hasproperty(sol, :stats) ? sol.stats : sol.destats
    (naccept = Int(getproperty(stats, :naccept)),
     nreject = Int(getproperty(stats, :nreject)),
     nf = Int(getproperty(stats, :nf)))
end

function run_solve(case, tol, horizon; warmup=false)
    p = with_current(case.params, case.Je)
    u0 = copy(case.u0)
    cache = TDGLThermalCache(p)
    prob = ODEProblem(fdm_rhs_compatible!, u0, (0.0, horizon), cache)
    Random.seed!(case.seed)
    started = time_ns()
    sol = solve(prob, Tsit5(); abstol=tol, reltol=tol, save_everystep=false,
                dense=false, save_start=false, save_end=true)
    seconds = (time_ns() - started) / 1e9
    return sol, seconds
end

function final_sanity(case, sol)
    p = with_current(case.params, case.Je)
    u = copy(sol.u[end])
    cache = TDGLThermalCache(p)
    du = similar(u)
    # Capture solver statistics before this separate sanity RHS call.  With
    # noise enabled, this diagnostic intentionally consumes RNG after solve.
    fdm_rhs_compatible!(du, u, cache, sol.t[end])
    vector_to_state!(cache.state, u, p)
    (final_V=compute_voltage_from_du(du, p),
     final_T_avg=mean(cache.state.T[2:end-1, 2:end-1]),
     final_psi2_avg=mean(abs2.(cache.state.psi[2:end-1, 2:end-1])))
end

raw_inputs = JSON3.read(read(INPUTS_PATH, String))
inputs = collect(raw_inputs)
1 <= length(inputs) <= 16 || error("--inputs must contain 1..16 archived states")

jobs = NamedTuple[]
for item in inputs
    hasproperty(item, :Je) && hasproperty(item, :initial_state_path) ||
        error("each input requires Je and initial_state_path")
    Je = Float64(item.Je)
    path = String(item.initial_state_path)
    isfile(path) || error("archived initial_state_path not found: $path")
    for noise in (0.0, 0.001), tolerance in (1e-3, 1e-4, 1e-5)
        case = make_case(CONFIG_PATH; Je=Je, noise=noise, seed=2026110411,
                         preparation=:configured_down, initial_state_path=path)
        if hasproperty(item, :initial_output_sha256)
            case.initial_source_hash == String(item.initial_output_sha256) ||
                error("archived source hash mismatch at Je=$Je")
        end
        push!(jobs, (; case, noise, tolerance, initial_state_path=path))
    end
end

for item in inputs
    Je = Float64(item.Je)
    pair = filter(j -> j.case.Je == Je, jobs)
    hashes = unique(j.case.initial_hash for j in pair)
    length(hashes) == 1 || error("noise cases changed initial hash at Je=$Je")
end

rows = NamedTuple[]
for job in jobs
    # One short, unrecorded solve per case excludes compilation from timing.
    run_solve(job.case, job.tolerance, min(2.0, HORIZON); warmup=true)
    sol, seconds = run_solve(job.case, job.tolerance, HORIZON)
    stats = solve_stats(sol)
    sanity = final_sanity(job.case, sol)
    finite = all(isfinite, sol.u[end])
    push!(rows, (; Je=job.case.Je, initial_state_path=job.initial_state_path,
                 initial_hash=job.case.initial_hash, noise=job.noise,
                 initial_source_hash=job.case.initial_source_hash,
                 tolerance=job.tolerance, status=string(sol.retcode),
                 finite, complete=string(sol.retcode) in ("Success", "ReturnCode.Success") && isapprox(Float64(sol.t[end]), HORIZON;
                                            atol=8eps(max(HORIZON, 1.0))) && finite,
                 final_time=Float64(sol.t[end]), seconds,
                 naccept=stats.naccept, nreject=stats.nreject, nf=stats.nf,
                 avg_h=stats.naccept > 0 ? HORIZON / stats.naccept : NaN,
                 sanity...))
    println(JSON3.write((Je=job.case.Je, noise=job.noise, tolerance=job.tolerance,
                         naccept=stats.naccept, nreject=stats.nreject)))
end

receipt = (; schema="tdgl-noise-timestep-probe-v1",
           created_utc=string(now(UTC)),
           scope="Simplified archived-state adaptive Tsit5 timing; no HDF/images/streaming diagnostics; not a noise-on/off physical-equivalence test or production speed claim.",
           protocol="same configured_down archived fields and seed 2026110411; fresh cache/u0 and reseeded RNG per solve; warmup horizon=min(2,horizon) excluded",
           config_path=abspath(CONFIG_PATH), config_hash=sha(CONFIG_PATH),
           model_hash=sha(MODEL_PATH), rhs_source_hash=sha(RHS_PATH),
           inputs_sha256=sha(INPUTS_PATH), probe_sha256=sha(@__FILE__),
           julia=string(VERSION), machine=Sys.MACHINE, horizon=HORIZON,
           tolerances=[1e-3, 1e-4, 1e-5], noises=[0.0, 0.001], rows)

mkpath(dirname(OUTPUT_PATH))
open(OUTPUT_PATH, "w") do io
    write(io, JSON3.write(receipt)); write(io, '\n')
end
println(JSON3.write((output=OUTPUT_PATH, rows=length(rows))))
