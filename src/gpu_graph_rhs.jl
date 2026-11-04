# Optional deterministic replay of the validated full-curl kernel pipeline.
# The production solver does not load this file.
using CUDA

mutable struct GPUFullCurlGraphRHS
    cache::GPUFullCurlCache
    graphs::Dict{Tuple{UInt,UInt},CUDA.CuGraphExec}
    max_graphs::Int
    capture_seconds::Float64
end

function GPUFullCurlGraphRHS(cache::GPUFullCurlCache; max_graphs=64)
    max_graphs > 0 || throw(ArgumentError("max_graphs must be positive"))
    GPUFullCurlGraphRHS(cache, Dict{Tuple{UInt,UInt},CUDA.CuGraphExec}(), max_graphs, 0.0)
end

function (rhs::GPUFullCurlGraphRHS)(du::CuArray{ComplexF64,2}, u::CuArray{ComplexF64,2}, p, t)
    # Graphs retain their exact input/output storage. Adaptive RK stages may
    # use different arrays, so retain one executable per buffer pair.
    key = (UInt(pointer(u)), UInt(pointer(du)))
    if !haskey(rhs.graphs, key)
        length(rhs.graphs) < rhs.max_graphs || error("GPU RHS graph storage limit reached")
        start = time_ns()
        gpu_full_curl_rhs!(du, u, rhs.cache, t)
        CUDA.synchronize()
        graph = CUDA.capture() do
            gpu_full_curl_rhs!(du, u, rhs.cache, t)
        end
        rhs.graphs[key] = CUDA.instantiate(graph)
        rhs.capture_seconds += (time_ns() - start) / 1e9
    end
    CUDA.launch(rhs.graphs[key])
    return nothing
end
