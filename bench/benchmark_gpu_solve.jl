# Deterministic, Float64 whole-solver GPU benchmark.  This is an opt-in
# experiment and does not benchmark production noise or full output pipelines.
using CUDA
using OrdinaryDiffEq
using JSON3
using SHA
using Statistics
using Dates
using LinearAlgebra

include(joinpath(@__DIR__, "fixtures.jl"))
if !isdefined(Main, :GPUFullCurlCache) || !isdefined(Main, :gpu_full_curl_rhs!)
    include(joinpath(@__DIR__, "..", "src", "gpu_full_curl.jl"))
end

const SOLVE_SCOPE = "deterministic-time-integration-with-common-time-state-saving"
const CURRENTS = (0.06, 0.18, 0.27)

_arg(name, default=nothing) = begin
    i = findfirst(==("--" * name), ARGS)
    i === nothing ? default : (i == length(ARGS) ? error("missing value for --$name") : ARGS[i + 1])
end

function _write_json(path, x)
    open(path, "w") do io; write(io, JSON3.write(x)); write(io, '\n'); end
end

function _resources()
    gpu=try read(`nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits`,String) catch; "unavailable" end
    (load_average=Sys.loadavg(),free_host_memory=Sys.free_memory(),gpu_free_used_MiB_utilization_percent=strip(gpu))
end

function _norm(a, b)
    maximum(abs, a .- b) / (1 + maximum(abs, b))
end

function _stats(sol)
    s = hasproperty(sol, :stats) ? sol.stats : sol.destats
    (nf=try getproperty(s, :nf) catch; nothing end,
     naccept=try getproperty(s, :naccept) catch; nothing end,
     nreject=try getproperty(s, :nreject) catch; nothing end,
     retcode=string(sol.retcode))
end

# SciML also calls internalnorm on individual elements while constructing
# scaled residuals. Keep that scalar method GPU-safe; only the complete
# matrix callback performs a device reduction across trajectory columns.
_batch_internalnorm(x::Number, t) = abs(x)
_batch_internalnorm(x::AbstractMatrix, t) = sqrt(maximum(sum(abs2, x; dims=1)) / size(x,1))
_batch_internalnorm(x::AbstractVector, t) = sqrt(sum(abs2, x) / length(x))

function _assert_solution(sol, times)
    string(sol.retcode) in ("Success","ReturnCode.Success") || error("solver failed: $(sol.retcode)")
    length(sol.t)==length(times) && all(isapprox.(sol.t,times;rtol=0,atol=1e-10)) ||
        error("solver did not reach every requested common time")
    return nothing
end

function _cases(cfg, batch, nx, ny, preparation)
    currents = [CURRENTS[mod1(b, 3)] for b in 1:batch]
    cases = [make_case(cfg; Je=currents[b], noise=0.0, seed=20261104 + b,
        preparation=preparation, Nx=nx, Ny=ny) for b in 1:batch]
    p = cases[1].params
    p.bc_x === :periodic && p.bc_y === :neumann && !p.loop && p.field_side === :both &&
        p.epsilon_type === :linear && !p.psi_y_zero || error("unsupported GPU solver geometry")
    return p, currents, cases
end

function _cpu_solve(case, tol, times; warmup=false)
    p = with_current(case.params, case.Je)
    u0 = copy(case.u0); cache = TDGLThermalCache(p)
    prob = ODEProblem(fdm_rhs_compatible!, u0, (times[1], times[end]), cache)
    warmup && solve(prob, Tsit5(); abstol=tol, reltol=tol, saveat=times, tstops=times,
        save_everystep=false, dense=false)
    t0 = time_ns()
    sol = solve(prob, Tsit5(); abstol=tol, reltol=tol, saveat=times, tstops=times,
        save_everystep=false, dense=false)
    seconds=(time_ns() - t0) / 1e9
    _assert_solution(sol,times)
    (solution=sol, seconds=seconds, stats=_stats(sol))
end

function _gpu_solve(p, currents, h_u, tol, times; warmup=false)
    setup_start=time_ns()
    d_u = CuArray(h_u); cache = GPUFullCurlCache(p, currents)
    CUDA.synchronize()
    setup_seconds=(time_ns()-setup_start)/1e9
    rhs! = (du, u, c, t) -> gpu_full_curl_rhs!(du, u, cache, t)
    prob = ODEProblem(rhs!, d_u, (times[1], times[end]), nothing)
    kwargs = (; abstol=tol, reltol=tol, saveat=times, tstops=times, save_everystep=false,
        dense=false,internalnorm=_batch_internalnorm)
    warmup && (solve(prob, Tsit5(); kwargs...); CUDA.synchronize())
    CUDA.synchronize(); t0 = time_ns()
    sol = solve(prob, Tsit5(); kwargs...)
    CUDA.synchronize()
    seconds=(time_ns()-t0)/1e9
    _assert_solution(sol,times)
    (solution=sol, seconds=seconds, stats=_stats(sol),
     setup_seconds=setup_seconds,input_device_bytes=sizeof(eltype(h_u))*length(h_u))
end

function _field_ranges(p)
    nψ = (p.Nx + 1) * (p.Ny + 1); nA = p.Nx * (p.Ny + 1); nB = (p.Nx + 1) * p.Ny
    (psi=1:nψ, Ax=(nψ + 1):(nψ + nA), Ay=(nψ + nA + 1):(nψ + nA + nB),
     T=(nψ + nA + nB + 1):(nψ + nA + nB + nψ))
end

function _field_errors(states_a, states_b, p)
    rs = _field_ranges(p)
    NamedTuple{keys(rs)}(map(values(rs)) do r
        maximum(maximum(abs,states_a[k][r]-states_b[k][r]) for k in eachindex(states_a)) /
        (1+maximum(maximum(abs,states_b[k][r]) for k in eachindex(states_b)))
    end)
end

function _observables(states,p)
    c=TDGLThermalCache(p);du=similar(states[1])
    voltage=Float64[];temperature=Float64[];psi2=Float64[];heat=Matrix{Float64}[];magnetic=Matrix{Float64}[]
    for (k,u) in enumerate(states)
        fdm_rhs_compatible!(du,u,c,k-1.0)
        push!(voltage,real(compute_voltage_from_du(du,p)))
        push!(temperature,mean(c.state.T[2:end-1,2:end-1]))
        push!(psi2,mean(abs2,c.state.psi[2:end-1,2:end-1]))
        push!(heat,copy(c.W_total))
        ax=c.state.Ax;ay=c.state.Ay
        push!(magnetic,[(ay[i+1,j]-ay[i,j])/p.hx-(ax[i,j+1]-ax[i,j])/p.hy for i in 1:p.Nx,j in 1:p.Ny])
    end
    (;voltage,temperature,psi2,heat,magnetic)
end

function _observable_errors(a,b)
    one(x,y)=maximum(maximum(abs,x[k]-y[k]) for k in eachindex(x))/(1+maximum(maximum(abs,y[k]) for k in eachindex(y)))
    (voltage=_norm(a.voltage,b.voltage),temperature=_norm(a.temperature,b.temperature),psi2=_norm(a.psi2,b.psi2),
     heat=one(a.heat,b.heat),magnetic=one(a.magnetic,b.magnetic))
end

function _all_errors(a,b,p,ad,bd)
    merge(_field_errors(a,b,p),_observable_errors(ad,bd))
end

function _reference_check(case,cpu_states,gpu_states,times)
    p=with_current(case.params,case.Je)
    baseline=_cpu_solve(case,1e-4,times).solution.u
    levels=Any[];tolerances=Float64[];observables=Any[];converged=false
    residual=nothing;previous=nothing
    for tol in (1e-5,1e-6,1e-7,1e-8,1e-9)
        states=_cpu_solve(case,tol,times).solution.u
        push!(levels,states);push!(tolerances,tol);push!(observables,_observables(states,p))
        if length(levels)>=3
            j=length(levels)
            previous=_all_errors(levels[j-2],levels[j-1],p,observables[j-2],observables[j-1])
            residual=_all_errors(levels[j-1],levels[j],p,observables[j-1],observables[j])
            converged=all(residual[f]<=0.5*previous[f] || max(residual[f],previous[f])<=1e-12 for f in keys(residual))
            converged && break
        end
    end
    tight=levels[end];td=observables[end]
    base=_all_errors(baseline,tight,p,_observables(baseline,p),td)
    gpuerr=_all_errors(gpu_states,tight,p,_observables(gpu_states,p),td)
    cpuerr=_all_errors(cpu_states,tight,p,_observables(cpu_states,p),td)
    budget=NamedTuple{keys(base)}(map(f -> max(1.1*base[f],5*residual[f],1e-9),keys(base)))
    (current=case.Je,seed=case.seed,reference_tolerances=tolerances,previous_error=previous,residual_error=residual,
     reference_converged=converged,baseline_1e4_vs_reference=base,cpu_vs_reference=cpuerr,gpu_vs_reference=gpuerr,
     allowed_budget=budget,accurate=converged && all(gpuerr[f]<=budget[f] for f in keys(base)))
end

function _state_checks(v, p)
    nψ = (p.Nx + 1) * (p.Ny + 1); nA = p.Nx * (p.Ny + 1); nB = (p.Nx + 1) * p.Ny
    rax = (nψ + 1):(nψ + nA); ray = (nψ + nA + 1):(nψ + nA + nB)
    rt = (nψ + nA + nB + 1):(nψ + nA + nB + nψ)
    (finite=all(isfinite, v), max_imag_A=max(maximum(abs, imag.(v[rax])), maximum(abs, imag.(v[ray]))),
     max_imag_T=maximum(abs, imag.(v[rt])),min_temperature=minimum(real,v[rt]),max_temperature=maximum(real,v[rt]))
end

function _run_batch(cfg, batch, nx, ny, horizon, tol, repeats, preparation)
    p, currents, cases = _cases(cfg, batch, nx, ny, preparation)
    times = collect(0.0:1.0:horizon); times[end] == horizon || push!(times, horizon)
    n = length(cases[1].u0); h_u = Matrix{ComplexF64}(undef, n, batch)
    for b in 1:batch; h_u[:, b] .= cases[b].u0; end
    # One complete 2τ warmup per backend, outside all samples.
    warm_horizon=min(2.0,horizon)
    warm_times = collect(0.0:1.0:warm_horizon); warm_times[end] == warm_horizon || push!(warm_times,warm_horizon)
    _cpu_solve(cases[1],tol,warm_times)
    _gpu_solve(p,currents,h_u,tol,warm_times)
    cpu_samples = Any[]; gpu_samples = Any[]; cpu_solutions = Any[]; gpu_solutions = Any[]
    for rep in 1:repeats
        resource_before=_resources()
        if isodd(rep)
            c = [_cpu_solve(cases[b], tol, times; warmup=false) for b in 1:batch]
            push!(cpu_samples, (seconds=sum(x.seconds for x in c), stats=[x.stats for x in c])); cpu_solutions = c
            g = _gpu_solve(p, currents, h_u, tol, times; warmup=false)
        else
            g = _gpu_solve(p, currents, h_u, tol, times; warmup=false)
            c = [_cpu_solve(cases[b], tol, times; warmup=false) for b in 1:batch]
            push!(cpu_samples, (seconds=sum(x.seconds for x in c), stats=[x.stats for x in c])); cpu_solutions = c
        end
        push!(gpu_samples, (seconds=g.seconds, stats=g.stats, setup_seconds=g.setup_seconds,input_device_bytes=g.input_device_bytes,
            resource_before=resource_before,resource_after=_resources())); gpu_solutions = g
    end
    gfinal = gpu_solutions.solution
    gpu_states = [Array(gfinal.u[k]) for k in eachindex(gfinal.u)]
    cpu_states = [[Vector(cpu_solutions[b].solution.u[k]) for k in eachindex(cpu_solutions[b].solution.u)] for b in 1:batch]
    checks = [_state_checks(gpu_states[k][:, b], p) for b in 1:batch, k in eachindex(times)]
    # Tight references for the three selected currents establish the CPU
    # tolerance ladder used for the achieved-error budget.
    ref = Any[]
    for b in unique([findfirst(==(j), currents) for j in CURRENTS if any(==(j), currents)])
        check=_reference_check(cases[b],cpu_states[b],[gpu_states[k][:,b] for k in eachindex(times)],times)
        push!(ref,merge((batch_index=b,),check))
    end
    (batch=batch, currents=currents,initial_hashes=[c.initial_hash for c in cases], horizon=horizon, save_count=length(times), tolerance=tol,
     cpu_samples=cpu_samples, cpu_median=median(getproperty.(cpu_samples, :seconds)),
     gpu_samples=gpu_samples, gpu_median=median(getproperty.(gpu_samples, :seconds)),
     references=ref, reference_scope="first matched state of each current only",accurate=all(r.accurate for r in ref),
     all_column_errors_vs_independent_cpu=[_field_errors([gpu_states[k][:,b] for k in eachindex(times)],cpu_states[b],p) for b in 1:batch],
     checks=(all_finite=all(c -> c.finite, checks), max_imag_A=maximum(c -> c.max_imag_A, checks),
         max_imag_T=maximum(c -> c.max_imag_T, checks),min_temperature=minimum(c.min_temperature for c in checks),
         max_temperature=maximum(c.max_temperature for c in checks)), gpu_final_stats=_stats(gfinal))
end

function run_benchmark(; out, config, batches=[1, 8, 32, 99], repetitions=3,
                       horizon=100.0, tolerance=1e-6, nx=60, ny=30, preparation=:cool, noise=0.0)
    noise == 0.0 || error("GPU whole-solver benchmark rejects noise != 0")
    horizon>0 && tolerance>0 && repetitions>0 && all(b -> b>0,batches) || error("invalid positive benchmark setting")
    estimated_maximum_device_bytes=16*((nx+1)*(ny+1)*2+nx*(ny+1)+(nx+1)*ny)*maximum(batches)*(ceil(Int,horizon)+55)*3
    estimated_maximum_device_bytes<8*1024^3 || error("estimated device memory exceeds experiment budget")
    CUDA.functional() || error("CUDA.functional() is false")
    CUDA.allowscalar(false); mkpath(out)
    rows = [_run_batch(config, b, nx, ny, horizon, tolerance, repetitions, preparation) for b in batches]
    manifest = (schema_version="tdgl-gpu-solver-benchmark-v1", scope=SOLVE_SCOPE,
        deterministic=true, production_noise_claim=false, full_output_claim=false,
        precision="ComplexF64/Float64", config_path=abspath(config),
        config_hash=bytes2hex(SHA.sha256(read(config))), model_hash=bytes2hex(SHA.sha256(read(BENCH_MODEL_FILE))),
        gpu_source_hash=bytes2hex(SHA.sha256(read(joinpath(@__DIR__, "..", "src", "gpu_full_curl.jl")))),
        cpu_source_hash=bytes2hex(SHA.sha256(read(joinpath(@__DIR__, "..", "src", "cpu_compatible.jl")))),
        benchmark_source_hash=bytes2hex(SHA.sha256(read(@__FILE__))),
        environment_manifest_hash=bytes2hex(SHA.sha256(read(joinpath(dirname(Base.active_project()),"Manifest.toml")))),
        julia_version=string(VERSION), machine=Sys.MACHINE, created_utc=string(now(UTC)),
        horizon=horizon, tolerance=tolerance, repetitions=repetitions, Nx=nx, Ny=ny,
        rows=rows, transfers="excluded from solve timing; final states copied after timing",
        compilation="excluded by 2-time-unit warmup", internalnorm="maximum per-column RMS",estimated_maximum_device_bytes)
    _write_json(joinpath(out, "gpu_solver_benchmark.json"), manifest); manifest
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = _arg("out"); out === nothing && error("--out is required")
    cfg = _arg("config", joinpath(@__DIR__, "..", "config.yaml"))
    grid = split(_arg("grid", "60x30"), 'x'); length(grid) == 2 || error("--grid must be Nx x Ny")
    run_benchmark(out=String(out), config=String(cfg),
        batches=parse.(Int, split(_arg("batches", "1,8,32,99"), ',')),
        repetitions=parse(Int, _arg("repetitions", "3")), horizon=parse(Float64, _arg("horizon", "100")),
        tolerance=parse(Float64, _arg("tolerance", "1e-6")), nx=parse(Int, grid[1]), ny=parse(Int, grid[2]),
        preparation=Symbol(_arg("preparation", "cool")), noise=parse(Float64, _arg("noise", "0")))
end
