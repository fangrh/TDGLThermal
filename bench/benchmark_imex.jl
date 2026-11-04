"""Deterministic-only Tsit5 versus coupled sparse IMEX experiment.

This benchmark is opt-in and never changes the production backend.  Assembly
and solve timings are reported separately; all methods use the same packed
initial state and integer save/tstop grid.
"""

using Random, JSON3, Statistics, Dates, Printf
using OrdinaryDiffEq
using OrdinaryDiffEqSDIRK: KenCarp4
using OrdinaryDiffEqStabilizedRK: ROCK4
using SciMLOperators: MatrixOperator
include(joinpath(@__DIR__, "fixtures.jl"))
if !isdefined(Main, :fdm_rhs_compatible!)
    include(joinpath(@__DIR__, "..", "src", "cpu_compatible.jl"))
end
if !isdefined(Main, :build_spatial_split)
    include(joinpath(@__DIR__, "..", "src", "imex_experiment.jl"))
end

_count(s, name) = hasproperty(s, name) ? getproperty(s, name) : missing
_solution_stats(sol) = hasproperty(sol, :stats) ? sol.stats : sol.destats

function _run_tsit5(case, horizon, tol, times; warmup=true)
    p = with_current(case.params, case.Je)
    u0 = copy(case.u0); cache = TDGLThermalCache(p); du = similar(u0)
    warmup && solve(ODEProblem(fdm_rhs_compatible!, u0, (0., horizon), cache), Tsit5();
                    abstol=tol, reltol=tol, saveat=times, tstops=times,
                    save_everystep=false)
    sol = @timed solve(ODEProblem(fdm_rhs_compatible!, u0, (0., horizon), cache), Tsit5();
                       abstol=tol, reltol=tol, saveat=times, tstops=times,
                       save_everystep=false)
    s = _solution_stats(sol.value)
    (solver="Tsit5", wall_seconds=sol.time, final_time=sol.value.t[end],
     status=string(sol.value.retcode), accepted_steps=_count(s, :naccept),
     rejected_steps=_count(s, :nreject), rhs_calls=_count(s, :nf),
     states=sol.value.u, times=sol.value.t, final_state=sol.value.u[end])
end

function _run_kencarp(case, split, horizon, tol, times; warmup=true)
    p = with_current(case.params, case.Je)
    # The reference cache is separate from assembly scratch/state.
    ctx = ImexContext(split; reference_cache=TDGLThermalCache(p))
    u0 = copy(case.u0)
    # SciML's current SplitFunction requires a SciMLOperators MatrixOperator;
    # passing the raw sparse matrix is rejected during isinplace detection.
    prob = SplitODEProblem(MatrixOperator(ctx.L), imex_residual!, u0,
                           (0., horizon), ctx)
    warmup && solve(prob, KenCarp4(); abstol=tol, reltol=tol, saveat=times, tstops=times,
                    save_everystep=false)
    timed = @timed solve(prob, KenCarp4(); abstol=tol, reltol=tol,
                         saveat=times, tstops=times, save_everystep=false)
    s = _solution_stats(timed.value)
    (solver="KenCarp4", wall_seconds=timed.time, final_time=timed.value.t[end],
     status=string(timed.value.retcode), accepted_steps=_count(s, :naccept),
     rejected_steps=_count(s, :nreject), rhs_calls=_count(s, :nf),
     states=timed.value.u, times=timed.value.t, final_state=timed.value.u[end])
end

function _run_rock4(case, horizon, tol, times; warmup=true)
    p = with_current(case.params, case.Je)
    u0 = copy(case.u0); cache = TDGLThermalCache(p)
    prob = ODEProblem(fdm_rhs_compatible!, u0, (0., horizon), cache)
    warmup && solve(prob, ROCK4(); abstol=tol, reltol=tol, saveat=times,
                    tstops=times, save_everystep=false)
    timed = @timed solve(prob, ROCK4(); abstol=tol, reltol=tol, saveat=times,
                         tstops=times, save_everystep=false)
    s = _solution_stats(timed.value)
    (solver="ROCK4", wall_seconds=timed.time, final_time=timed.value.t[end],
     status=string(timed.value.retcode), accepted_steps=_count(s, :naccept),
     rejected_steps=_count(s, :nreject), rhs_calls=_count(s, :nf),
     states=timed.value.u, times=timed.value.t, final_state=timed.value.u[end])
end

function _field_errors(a, b, p)
    nψ = (p.Nx + 1) * (p.Ny + 1); nAx = p.Nx * (p.Ny + 1); nAy = (p.Nx + 1) * p.Ny
    ranges = (1:nψ, nψ+1:nψ+nAx, nψ+nAx+1:nψ+nAx+nAy, nψ+nAx+nAy+1:length(a))
    map(r -> maximum(abs, a[r] .- b[r]) / (1 + maximum(abs, b[r])), ranges)
end

function _diagnostics(states, p)
    cache = TDGLThermalCache(p); du = zeros(ComplexF64, length(states[1])); rows = NamedTuple[]
    for (t, u) in enumerate(states)
        fdm_rhs_compatible!(du, u, cache, t - 1.)
        nψ = (p.Nx + 1) * (p.Ny + 1); nAx = p.Nx * (p.Ny + 1); nAy = (p.Nx + 1) * p.Ny
        astate = u[nψ+1:nψ+nAx+nAy]
        tstate = u[nψ+nAx+nAy+1:end]
        heat = cache.W_total
        push!(rows, (voltage=real(compute_voltage_from_du(du,p)), temperature=mean(cache.state.T[2:end-1,2:end-1]),
            heat=maximum(abs, heat), imag_A=maximum(abs, imag.(astate)),
            imag_T=maximum(abs, imag.(tstate))))
    end
    rows
end

function run_imex_case(case; horizon::Float64=100., tolerance::Float64=1e-6,
                       repetitions::Int=1,
                       solvers::Vector{Symbol}=[:Tsit5,:KenCarp4,:ROCK4])
    case.noise == 0.0 || throw(ArgumentError("IMEX experiment is deterministic-only"))
    times = collect(0.:1.:horizon)
    times[end] == horizon || push!(times,horizon)
    p = with_current(case.params, case.Je)
    all(s -> s in (:Tsit5,:KenCarp4,:ROCK4),solvers) || error("unknown solver")
    assembly = :KenCarp4 in solvers ? (@timed build_spatial_split(p)) :
        (value=nothing,time=0.0,bytes=0)
    split = assembly.value
    rows = NamedTuple[]
    # Tightened Tsit5 is the deterministic reference for this small experiment.
    reference5 = _run_tsit5(case,horizon,1e-5,times;warmup=true)
    reference6 = _run_tsit5(case,horizon,1e-6,times;warmup=false)
    reference = _run_tsit5(case,horizon,1e-7,times;warmup=false)
    trajectory_errors(a,b) = reduce((x,y)->max.(x,y),
        (_field_errors(a.states[i],b.states[i],p) for i in eachindex(b.states));init=zeros(4))
    residual_error = trajectory_errors(reference6,reference)
    previous_error = trajectory_errors(reference5,reference6)
    reference_converged = all((residual_error .<= 0.5 .* previous_error) .|
        ((residual_error .<= 1e-12) .& (previous_error .<= 1e-12)))
    reference_tolerance = 1e-7
    for tighter in (1e-8,1e-9)
        reference_converged && break
        reference5,reference6 = reference6,reference
        reference = _run_tsit5(case,horizon,tighter,times;warmup=false)
        residual_error = trajectory_errors(reference6,reference)
        previous_error = trajectory_errors(reference5,reference6)
        reference_converged = all((residual_error .<= 0.5 .* previous_error) .|
            ((residual_error .<= 1e-12) .& (previous_error .<= 1e-12)))
        reference_tolerance = tighter
    end
    baseline = _run_tsit5(case, horizon, 1e-4, times; warmup=true)
    :KenCarp4 in solvers && _run_kencarp(case, split, horizon, tolerance, times; warmup=true)
    :ROCK4 in solvers && _run_rock4(case, horizon, tolerance, times; warmup=true)
    pdiag = with_current(case.params, case.Je)
    refdiag = _diagnostics(reference.states, pdiag)
    basediag = _diagnostics(baseline.states, pdiag)
    for rep in 1:repetitions
        order = isodd(rep) ? solvers : reverse(solvers)
        for solver in order
            result = solver === :Tsit5 ? _run_tsit5(case, horizon, tolerance, times; warmup=false) :
                solver === :KenCarp4 ? _run_kencarp(case, split, horizon, tolerance, times; warmup=false) :
                _run_rock4(case, horizon, tolerance, times; warmup=false)
            ferr = map(i -> _field_errors(result.states[i], reference.states[i], pdiag), eachindex(result.states))
            maxerr = reduce((a, b) -> max.(a, b), ferr; init=zeros(4))
            refres = map(i -> _field_errors(baseline.states[i], reference.states[i], pdiag), eachindex(reference.states))
            refmax = reduce((a, b) -> max.(a, b), refres; init=zeros(4))
            budget = max.(1.1 .* refmax, 5 .* residual_error, 1e-9)
            diag = _diagnostics(result.states, pdiag)
            maxdiag = maximum(abs, [d.temperature - refdiag[i].temperature for (i,d) in enumerate(diag)])
            converged = reference_converged && all(maxerr .≤ budget)
            push!(rows, (; repetition=rep, Je=case.Je, preparation=string(case.preparation),
                tolerance, horizon, assembly_seconds=assembly.time,
                solver=result.solver, wall_seconds=result.wall_seconds,
                final_time=result.final_time, status=result.status,
                accepted_steps=result.accepted_steps, rejected_steps=result.rejected_steps,
                rhs_calls=result.rhs_calls,
                field_errors=maxerr, approved_error_budget=budget,
                diagnostic_temperature_error=maxdiag,
                voltage=diag[end].voltage, temperature=diag[end].temperature,
                heat=diag[end].heat, max_imag_A=maximum(d.imag_A for d in diag),
                max_imag_T=maximum(d.imag_T for d in diag),
                finite=all(isfinite, result.final_state), converged=converged,
                status_ok=(result.status == "Success" && converged)))
        end
    end
    (; schema_version="tdgl-imex-experiment-v1", deterministic_only=true,
       noise=case.noise, reference_converged, reference_tolerance, residual_error,
       assembly_seconds=assembly.time, assembly_bytes=assembly.bytes,
       rows)
end

function _arg(name, default=nothing)
    i = findfirst(==("--" * name), ARGS)
    i === nothing ? default : (i == length(ARGS) ? error("missing value for --$name") : ARGS[i+1])
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = _arg("out"); out === nothing && error("--out is required")
    cfg = _arg("config", joinpath(@__DIR__, "..", "config.yaml"))
    Je = parse(Float64, _arg("current", "0.18"))
    horizon = parse(Float64, _arg("horizon", "100"))
    tol = parse(Float64, _arg("tolerance", "1e-6"))
    reps = parse(Int, _arg("repetitions", "1"))
    case = make_case(cfg; Je=Je, noise=0., seed=20261104,
                     preparation=Symbol(_arg("preparation", "cool")),
                     Nx=parse(Int, _arg("nx", "60")), Ny=parse(Int, _arg("ny", "30")))
    solvers = Symbol.(Base.split(_arg("solvers","Tsit5,KenCarp4,ROCK4"),','))
    result = run_imex_case(case; horizon=horizon, tolerance=tol, repetitions=reps,solvers=solvers)
    mkpath(dirname(out))
    open(out, "w") do io
        JSON3.write(io, result); write(io, '\n')
    end
    @info "deterministic IMEX benchmark complete" out assembly_seconds=result.assembly_seconds
end
