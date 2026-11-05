# Exact replay and launch-overhead experiment; not a solver speed claim.
using CUDA, JSON3, Statistics, SHA
include(joinpath(@__DIR__, "benchmark_gpu_rhs.jl"))
include(joinpath(@__DIR__, "..", "src", "gpu_graph_rhs.jl"))

function graph_probe(out; batches=[1,8,32,99], repeats=100, repetitions=5)
    CUDA.allowscalar(false)
    cfg = joinpath(@__DIR__, "..", "config.yaml")
    rows = []
    for batch in batches
        p, currents, cases, hu = _make_batch(cfg, batch, 60, 30)
        u = CuArray(hu); plain_du=similar(u); graph_du=similar(u)
        c = GPUFullCurlCache(p, currents); gc = GPUFullCurlCache(p, currents)
        rhs = GPUFullCurlGraphRHS(gc)
        gpu_full_curl_rhs!(plain_du,u,c,0.0); rhs(graph_du,u,nothing,0.0); CUDA.synchronize()
        err = maximum(abs, Array(graph_du).-Array(plain_du))/(1+maximum(abs,Array(plain_du)))
        err <= 1e-11 || error("graph replay differs from ordinary RHS")
        # Replay must read changed contents rather than a captured host fixture.
        changed=0.95 .* hu
        copyto!(u,changed)
        gpu_full_curl_rhs!(plain_du,u,c,1.0); rhs(graph_du,u,nothing,1.0); CUDA.synchronize()
        err_changed=maximum(abs,Array(graph_du).-Array(plain_du))/(1+maximum(abs,Array(plain_du)))
        err_changed <= 1e-11 || error("graph replay uses stale state")
        all(Array(u).==changed) || error("graph RHS changed its input")
        other_u=CuArray(hu); other_du=similar(other_u)
        rhs(other_du,other_u,nothing,2.0); CUDA.synchronize()
        length(rhs.graphs)==2 || error("graph key is not buffer-specific")
        samples=[]
        for rep in 1:repetitions
            times=Dict{String,Float64}()
            for mode in (isodd(rep) ? ("plain","graph") : ("graph","plain"))
                CUDA.synchronize()
                sec=@elapsed begin
                    for _ in 1:repeats
                        mode=="plain" ? gpu_full_curl_rhs!(plain_du,u,c,0.0) : rhs(graph_du,u,nothing,0.0)
                    end
                    CUDA.synchronize()
                end
                times[mode]=sec/repeats
            end
            push!(samples,(plain_seconds=times["plain"],graph_seconds=times["graph"],speedup=times["plain"]/times["graph"]))
        end
        push!(rows,(batch=batch,errors=(initial=err,changed_state=err_changed),graphs=length(rhs.graphs),
            capture_seconds=rhs.capture_seconds,samples=samples,median_speedup=median(x.speedup for x in samples)))
    end
    mkpath(dirname(out))
    _write_json(out,(scope="deterministic-GPU-RHS-graph-only",rows=rows,julia_version=string(VERSION),
        graph_source_hash=bytes2hex(sha256(read(joinpath(@__DIR__,"..","src","gpu_graph_rhs.jl")))),
        source="https://cuda.juliagpu.org/stable/usage/graphs/"))
end

if abspath(PROGRAM_FILE)==@__FILE__
    graph_probe(_arg("out", "gpu_graph_probe.json"))
end
