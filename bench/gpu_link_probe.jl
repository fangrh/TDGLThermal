# GPU link-kernel feasibility probe. This is a link-kernel-only benchmark;
# it is not an end-to-end TDGL performance claim.
using CUDA
using JSON3
using Statistics
using Random
using SHA

const GRID_NX = 60
const GRID_NY = 30
const HX = 0.5
const HY = 0.5
const HOST_SEED = 20261104
const DEFAULT_REPEATED_CALLS = 100
const DEFAULT_SAMPLES = 5
const DEFAULT_THREADS = 256

"""Reference implementation of the compatible link operation."""
function cpu_links!(Ux, Uy, Ax, Ay, hx, hy)
    @inbounds for i in eachindex(Ax)
        Ux[i] = cis(-hx * Ax[i])
    end
    @inbounds for i in eachindex(Ay)
        Uy[i] = cis(-hy * Ay[i])
    end
    return nothing
end

"""One flat kernel handles both link arrays, whose lengths differ by one row."""
function gpu_links_kernel!(Ux, Uy, Ax, Ay, hx, hy)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Ax)
        Ux[i] = cis(-hx * Ax[i])
    end
    if i <= length(Ay)
        Uy[i] = cis(-hy * Ay[i])
    end
    return
end

function gpu_links!(Ux, Uy, Ax, Ay, hx, hy; threads=DEFAULT_THREADS)
    n = max(length(Ax), length(Ay))
    blocks = cld(n, threads)
    CUDA.@cuda threads=threads blocks=blocks gpu_links_kernel!(Ux, Uy, Ax, Ay, hx, hy)
    return nothing
end

function _arg(name, default=nothing)
    prefix = "--" * name
    for (i, arg) in pairs(ARGS)
        arg == prefix && return i == length(ARGS) ? error("missing value for $prefix") : ARGS[i + 1]
        startswith(arg, prefix * "=") && return split(arg, "="; limit=2)[2]
    end
    return default
end

function _cuda_string(name::Symbol, fallback="unavailable")
    try
        value = getproperty(CUDA, name)
        return string(value isa Function ? value() : value)
    catch
        return fallback
    end
end

function _memory_info()
    try
        free, total = CUDA.memory_info()
        return (free_bytes=Int(free), total_bytes=Int(total))
    catch
        return (free_bytes=nothing, total_bytes=nothing)
    end
end

function _write_json(path, value)
    parent = dirname(abspath(path))
    isempty(parent) || mkpath(parent)
    open(path, "w") do io
        write(io, JSON3.write(value))
        write(io, '\n')
    end
end

function _max_error(a, b)
    length(a) == length(b) || error("link arrays have different lengths")
    maximum((abs(a[i] - b[i]) for i in eachindex(a, b)); init=0.0)
end

function _timed_cpu(Ux, Uy, Ax, Ay, calls)
    elapsed = @elapsed begin
        for _ in 1:calls
            cpu_links!(Ux, Uy, Ax, Ay, HX, HY)
        end
    end
    return elapsed / calls
end

function _timed_gpu(Ux, Uy, Ax, Ay, calls, threads)
    elapsed = @elapsed begin
        CUDA.@sync begin
            for _ in 1:calls
                gpu_links!(Ux, Uy, Ax, Ay, HX, HY; threads=threads)
            end
        end
    end
    return elapsed / calls
end

function _timed_upload!(dAx, dAy, Ax, Ay, calls)
    elapsed = @elapsed begin
        CUDA.@sync begin
            for _ in 1:calls
                copyto!(dAx, Ax)
                copyto!(dAy, Ay)
            end
        end
    end
    return elapsed / calls
end

function _timed_download!(Ux, Uy, dUx, dUy, calls)
    elapsed = @elapsed begin
        CUDA.@sync begin
            for _ in 1:calls
                copyto!(Ux, dUx)
                copyto!(Uy, dUy)
            end
        end
    end
    return elapsed / calls
end

function _fixture(rng, batch)
    # The batch is deliberately the final axis, matching benchmark field output.
    Ax = rand(rng, Float64, GRID_NX, GRID_NY + 1, batch)
    Ay = rand(rng, Float64, GRID_NX + 1, GRID_NY, batch)
    Ux = Array{ComplexF64}(undef, size(Ax))
    Uy = Array{ComplexF64}(undef, size(Ay))
    return Ax, Ay, Ux, Uy
end

function _run_batch(batch; samples, calls, threads, rng)
    Ax, Ay, Ux, Uy = _fixture(rng, batch)
    dAx, dAy = CuArray(Ax), CuArray(Ay)
    dUx = similar(dAx, ComplexF64)
    dUy = similar(dAy, ComplexF64)
    # Compile and synchronize outside the measured samples.
    cpu_links!(Ux, Uy, Ax, Ay, HX, HY)
    gpu_links!(dUx, dUy, dAx, dAy, HX, HY; threads=threads)
    CUDA.synchronize()
    copyto!(Ux, dUx); copyto!(Uy, dUy)
    # Recompute the independent CPU reference after the warmup copy.
    cpu_links!(Ux, Uy, Ax, Ay, HX, HY)
    refUx, refUy = copy(Ux), copy(Uy)
    gpu_links!(dUx, dUy, dAx, dAy, HX, HY; threads=threads)
    CUDA.synchronize()
    copyto!(Ux, dUx); copyto!(Uy, dUy)
    gpu_error = max(_max_error(Ux, refUx), _max_error(Uy, refUy))
    gpu_error <= 1e-11 || error("GPU link validation failed for batch $batch: $gpu_error")

    cpu_seconds = Float64[]
    gpu_seconds = Float64[]
    upload_seconds = Float64[]
    download_seconds = Float64[]
    for _ in 1:samples
        push!(cpu_seconds, _timed_cpu(Ux, Uy, Ax, Ay, calls))
        push!(gpu_seconds, _timed_gpu(dUx, dUy, dAx, dAy, calls, threads))
        push!(upload_seconds, _timed_upload!(dAx, dAy, Ax, Ay, calls))
        push!(download_seconds, _timed_download!(Ux, Uy, dUx, dUy, calls))
    end
    entries = length(Ax) + length(Ay)
    per_call = (cpu=mean(cpu_seconds), gpu=mean(gpu_seconds),
                upload=mean(upload_seconds), download=mean(download_seconds))
    return (batch=batch, shapes=(Ax=collect(size(Ax)), Ay=collect(size(Ay))),
            entries_per_call=entries, repeated_calls=calls, samples=samples,
            max_error=gpu_error, cpu_seconds=cpu_seconds, gpu_seconds=gpu_seconds,
            upload_seconds=upload_seconds, download_seconds=download_seconds,
            per_call_seconds=per_call,
            entries_per_second=(cpu=entries / per_call.cpu, gpu=entries / per_call.gpu))
end

function main()
    out = _arg("out")
    out === nothing && error("--out is required")
    calls = parse(Int, _arg("repeated-calls", string(DEFAULT_REPEATED_CALLS)))
    samples = parse(Int, _arg("samples", string(DEFAULT_SAMPLES)))
    threads = parse(Int, _arg("threads", string(DEFAULT_THREADS)))
    calls > 0 && samples >= 5 && threads > 0 || error("invalid calls, samples, or threads")
    batches = parse.(Int, split(_arg("batches", "1,8,32,99"), ','))
    all(>(0), batches) || error("batches must be positive")
    rng = MersenneTwister(HOST_SEED)
    memory = _memory_info()
    rows = [_run_batch(batch; samples=samples, calls=calls, threads=threads, rng=rng) for batch in batches]
    source_hash = bytes2hex(SHA.sha256(read(@__FILE__)))
    result = (schema_version="gpu-link-probe-v1", scope="link-kernel-only",
              precision="Float64/ComplexF64", host_seed=HOST_SEED,
              grid=(Nx=GRID_NX, Ny=GRID_NY), spacings=(hx=HX, hy=HY),
              batches=batches, default_threads=DEFAULT_THREADS, threads=threads,
              repeated_calls=calls, samples=samples,
              device=_cuda_string(:device), runtime=_cuda_string(:runtime_version),
              driver=_cuda_string(:driver_version), memory=memory,
              julia_version=string(VERSION), machine=string(Sys.MACHINE),
              source_hash=source_hash, max_error=maximum(row.max_error for row in rows),
              results=rows)
    _write_json(out, result)
    println("wrote ", abspath(out))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
