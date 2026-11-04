# Device-resident full-curl RHS benchmark.  This is an opt-in deterministic
# benchmark; it does not run a solver or make a whole-solver speed claim.
using CUDA
using JSON3
using SHA
using Statistics
using Dates
using Printf
using Serialization

include(joinpath(@__DIR__, "fixtures.jl"))
if !isdefined(Main, :fdm_rhs_compatible!)
    include(joinpath(@__DIR__, "..", "src", "cpu_compatible.jl"))
end
if !isdefined(Main, :GPUFullCurlCache) || !isdefined(Main, :gpu_full_curl_rhs!)
    include(joinpath(@__DIR__, "..", "src", "gpu_full_curl.jl"))
end

const GPU_RHS_SCOPE = "full-RHS-only"
const DEFAULT_CURRENTS = (0.06, 0.18, 0.27)

_arg(name, default=nothing) = begin
    i = findfirst(==("--" * name), ARGS)
    i === nothing ? default : (i == length(ARGS) ? error("missing value for --$name") : ARGS[i + 1])
end

function _hash_bytes(x)
    io = IOBuffer(); serialize(io, x); bytes2hex(SHA.sha256(take!(io)))
end

function _write_json(path, x)
    open(path, "w") do io
        write(io, JSON3.write(x)); write(io, '\n')
    end
end

function _cuda_receipt()
    dev = try string(CUDA.device()) catch e "unavailable: $(sprint(showerror, e))" end
    drv = try string(CUDA.runtime_version()) catch e "unavailable: $(sprint(showerror, e))" end
    free_mem = try CUDA.available_memory() catch e nothing end
    total_mem = try CUDA.total_memory() catch e nothing end
    cuda_julia = try string(CUDA.versioninfo()) catch e "unavailable: $(sprint(showerror, e))" end
    (device=dev, cuda_runtime=drv, available_memory=free_mem, total_memory=total_mem,
     cuda_julia=cuda_julia)
end

function _validate_options(p, noise)
    p.bc_x === :periodic || error("GPU benchmark requires bc_x=:periodic")
    p.bc_y === :neumann || error("GPU benchmark requires bc_y=:neumann")
    !p.loop || error("GPU benchmark requires loop=false")
    p.field_side === :both || error("GPU benchmark requires field_side=:both")
    p.epsilon_type === :linear || error("GPU benchmark requires epsilon_type=:linear")
    !p.psi_y_zero || error("GPU benchmark requires psi_y_zero=false")
    noise == 0.0 || error("GPU benchmark is deterministic and requires noise=0")
end

function _make_batch(cfg, batch, nx, ny; preparation=:cool)
    currents = [DEFAULT_CURRENTS[mod1(b, length(DEFAULT_CURRENTS))] for b in 1:batch]
    cases = [make_case(cfg; Je=currents[b], noise=0.0, seed=20261104 + b,
                       preparation=preparation, Nx=nx, Ny=ny) for b in 1:batch]
    p = cases[1].params
    all(c -> c.params.Nx == nx && c.params.Ny == ny, cases) || error("fixture grid mismatch")
    _validate_options(p, 0.0)
    n = length(cases[1].u0)
    u = Matrix{ComplexF64}(undef, n, batch)
    for b in 1:batch
        u[:, b] .= cases[b].u0
    end
    return p, currents, cases, u
end

function _cpu_batch!(du, u, caches, t)
    @inbounds for b in axes(u, 2)
        fdm_rhs_compatible!(view(du, :, b), view(u, :, b), caches[b], t)
    end
    du
end

function _errors(cpu, gpu, p)
    np = (p.Nx + 1) * (p.Ny + 1)
    na = p.Nx * (p.Ny + 1)
    nb = (p.Nx + 1) * p.Ny
    ranges = (psi=1:np, Ax=(np + 1):(np + na), Ay=(np + na + 1):(np + na + nb),
              T=(np + na + nb + 1):(np + na + nb + np))
    (; psi=normalized_error(cpu[ranges.psi, :], gpu[ranges.psi, :]),
       Ax=normalized_error(cpu[ranges.Ax, :], gpu[ranges.Ax, :]),
       Ay=normalized_error(cpu[ranges.Ay, :], gpu[ranges.Ay, :]),
       T=normalized_error(cpu[ranges.T, :], gpu[ranges.T, :]),
       all=normalized_error(cpu, gpu))
end

function benchmark_batch_cpu!(du, u, caches, repeats=1)
    t0 = time_ns()
    for _ in 1:repeats
        _cpu_batch!(du, u, caches, 0.0)
    end
    (time_ns() - t0) / 1e9 / repeats
end

function _timed_gpu!(du, u, cache, repeats)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeats
        gpu_full_curl_rhs!(du, u, cache, 0.0)
    end
    CUDA.synchronize()
    (time_ns() - t0) / 1e9 / repeats
end

function run_benchmark(; out::String, config::String, batches=(1, 8, 32, 99),
                       repetitions=5, repeats=20, nx=60, ny=30, preparation=:cool)
    repetitions > 0 && repeats > 0 || error("repetitions and repeats must be positive")
    CUDA.functional() || error("CUDA.functional() is false; GPU benchmark requires a functional CUDA runtime")
    CUDA.allowscalar(false)
    mkpath(out)
    rows = Any[]
    for batch in batches
        p, currents, cases, h_u = _make_batch(config, batch, nx, ny; preparation=preparation)
        h_du = zeros(ComplexF64, size(h_u)); cpu_caches = [TDGLThermalCache(with_current(p, currents[b])) for b in 1:batch]
        free_before = try CUDA.available_memory() catch e nothing end
        d_u = CuArray(h_u); d_du = CUDA.zeros(ComplexF64, size(h_u)); gpu_cache = GPUFullCurlCache(p, currents)
        CUDA.synchronize()
        free_after = try CUDA.available_memory() catch e nothing end
        gpu_memory_delta = (free_before isa Integer && free_after isa Integer) ?
            max(Int128(0), Int128(free_before) - Int128(free_after)) : nothing
        gpu_memory_delta === nothing || gpu_memory_delta < 8_000_000_000 ||
            error("GPU benchmark resident allocation exceeds 8 GiB: $gpu_memory_delta bytes")
        # Compile/warm both paths outside timing, then validate every packed field
        # and verify that the input state was not modified.
        h_before = copy(h_u); d_before = Array(d_u)
        _cpu_batch!(h_du, h_u, cpu_caches, 0.0)
        gpu_full_curl_rhs!(d_du, d_u, gpu_cache, 0.0); CUDA.synchronize()
        h_gpu_du = Array(d_du)
        h_u == h_before || error("CPU RHS modified input state")
        Array(d_u) == d_before || error("GPU RHS modified input state")
        errs = _errors(h_du, h_gpu_du, p)
        all(errs[k] <= 1e-11 for k in (:psi, :Ax, :Ay, :T)) ||
            error("GPU/CPU fieldwise equivalence failed: $(errs)")
        samples = Any[]
        for rep in 1:repetitions
            # Paired order alternates to reduce systematic device/host bias.
            if isodd(rep)
                cpu_s = benchmark_batch_cpu!(h_du, h_u, cpu_caches, repeats)
                gpu_s = _timed_gpu!(d_du, d_u, gpu_cache, repeats)
            else
                gpu_s = _timed_gpu!(d_du, d_u, gpu_cache, repeats)
                cpu_s = benchmark_batch_cpu!(h_du, h_u, cpu_caches, repeats)
            end
            push!(samples, (cpu_seconds=cpu_s, gpu_seconds=gpu_s, speedup=cpu_s / gpu_s))
        end
        push!(rows, (batch=batch, Nx=nx, Ny=ny, currents=currents, repetitions=repetitions,
            repeats_per_sample=repeats, errors=errs, samples=samples,
            cpu_median=median(getproperty.(samples, :cpu_seconds)),
            gpu_median=median(getproperty.(samples, :gpu_seconds)),
            speedup_median=median(getproperty.(samples, :speedup)),
            resident_bytes=Base.summarysize(h_u) + Base.summarysize(h_du),
            gpu_memory_delta_bytes=gpu_memory_delta))
    end
    receipt = _cuda_receipt()
    manifest = (schema_version="tdgl-gpu-rhs-benchmark-v1", scope=GPU_RHS_SCOPE,
        deterministic=true, precision="ComplexF64/Float64", preparation=string(preparation),
        Nx=nx, Ny=ny, batches=collect(batches), repetitions=repetitions, repeats_per_sample=repeats,
        config_path=abspath(config), config_hash=bytes2hex(SHA.sha256(read(config))),
        model_hash=bytes2hex(SHA.sha256(read(BENCH_MODEL_FILE))),
        gpu_source_hash=isfile(joinpath(@__DIR__, "..", "src", "gpu_full_curl.jl")) ?
            bytes2hex(SHA.sha256(read(joinpath(@__DIR__, "..", "src", "gpu_full_curl.jl")))) : "missing",
        julia_version=string(VERSION), machine=Sys.MACHINE, created_utc=string(now(UTC)),
        cuda=receipt, rows=rows, transfer_timing="excluded; device arrays resident",
        allocation_timing="excluded; setup/allocation outside timed samples",
        memory_limit_bytes=8_000_000_000)
    _write_json(joinpath(out, "gpu_rhs_benchmark.json"), manifest)
    return manifest
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = _arg("out"); out === nothing && error("--out is required")
    cfg = _arg("config", joinpath(@__DIR__, "..", "config.yaml"))
    grid = split(_arg("grid", "60x30"), 'x'); length(grid) == 2 || error("--grid must be Nx x Ny")
    batches = parse.(Int, split(_arg("batches", "1,8,32,99"), ','))
    run_benchmark(out=String(out), config=String(cfg), batches=batches,
        repetitions=parse(Int, _arg("repetitions", "5")), repeats=parse(Int, _arg("repeats", "20")),
        nx=parse(Int, grid[1]), ny=parse(Int, grid[2]),
        preparation=Symbol(_arg("preparation", "cool")))
end
