# Complete-RHS test of equivalent KWT mobility arithmetic. Noise stays off.
include(joinpath(@__DIR__,"gpu_float32_probe.jl"))
include(joinpath(@__DIR__,"..","src","gpu_kwt_stable_experiment.jl"))

function run_kwt_probe(out;batches=[1,8,32,99],samples=5,repeats=20)
    CUDA.functional() || error("CUDA functional device required")
    CUDA.allowscalar(false);mkpath(out);rows=[]
    cfg=joinpath(@__DIR__,"..","config.yaml")
    for batch in batches
        p,currents,cases,hu=_make_batch(cfg,batch,60,30)
        cpu=similar(hu)
        for k in 1:batch
            fdm_rhs_compatible!(view(cpu,:,k),view(hu,:,k),TDGLThermalCache(with_current(p,currents[k])),0.)
        end
        u=CuArray(ComplexF32.(hu));c=GPUFullCurl32Cache(p,currents)
        du=[similar(u) for _ in 1:3]
        fs=(() -> gpu_full_curl32_rhs!(du[1],u,c,0.),
            () -> gpu_full_curl32_kwt_rhs!(du[2],u,c,0.;mode=:stable32),
            () -> gpu_full_curl32_kwt_rhs!(du[3],u,c,0.;mode=:promoted64))
        for f in fs;f();end
        CUDA.synchronize()
        errors=[_precision_errors(ComplexF64.(Array(d)),cpu,p) for d in du]
        Array(u)==ComplexF32.(hu) || error("RHS changed input")
        all(e -> all(isfinite,values(e)),errors) || error("nonfinite RHS errors")
        timings=[]
        for rep in 1:samples
            t=zeros(3)
            for k in (isodd(rep) ? (1,2,3) : (3,2,1))
                t[k]=_precision_time(fs[k],repeats)
            end
            push!(timings,(original=t[1],stable32=t[2],promoted64=t[3]))
        end
        push!(rows,(batch=batch,original_errors=errors[1],stable32_errors=errors[2],promoted64_errors=errors[3],samples=timings,
            original_median=median(x.original for x in timings),stable32_median=median(x.stable32 for x in timings),
            promoted64_median=median(x.promoted64 for x in timings)))
    end
    r=(scope="deterministic approximate KWT arithmetic, full-RHS-only",rows=rows,julia_version=string(VERSION),created_utc=string(now(UTC)),
        source_hash=bytes2hex(sha256(read(joinpath(@__DIR__,"..","src","gpu_kwt_stable_experiment.jl")))),
        benchmark_source_hash=bytes2hex(sha256(read(@__FILE__))),trajectory_claim=false,production_claim=false)
    _write_json(joinpath(out,"gpu_kwt_stable_probe.json"),r)
    r
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_kwt_probe(_arg("out","output/gpu_kwt_stable_probe"))
end
