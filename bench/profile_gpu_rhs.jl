# Profile one warmed, resident full RHS batch with CUDA.jl's integrated profiler.
include(joinpath(@__DIR__, "benchmark_gpu_rhs.jl"))
p,currents,cases,hu=_make_batch(joinpath(@__DIR__,"..","config.yaml"),99,60,30)
cu=CuArray(hu); du=similar(cu); cache=GPUFullCurlCache(p,currents)
gpu_full_curl_rhs!(du,cu,cache,0.0); CUDA.synchronize()
result=CUDA.@profile begin
    for _ in 1:1000
        gpu_full_curl_rhs!(du,cu,cache,0.0)
    end
    CUDA.synchronize()
end
display(result)
