# Lower-precision RHS feasibility; no trajectory/production accuracy claim.
using CUDA,JSON3,Dates,Statistics,Random,SHA
include(joinpath(@__DIR__,"benchmark_gpu_rhs.jl"))
include(joinpath(@__DIR__,"..","src","gpu_float32_experiment.jl"))

function _precision_errors(a,b,p)
    np=(p.Nx+1)*(p.Ny+1);na=p.Nx*(p.Ny+1);nb=(p.Nx+1)*p.Ny
    ranges=(psi=1:np,Ax=np+1:np+na,Ay=np+na+1:np+na+nb,T=np+na+nb+1:2np+na+nb)
    NamedTuple{keys(ranges)}(map(values(ranges)) do r
        maximum(abs,a[r,:]-b[r,:])/(1+maximum(abs,b[r,:]))
    end)
end

function _precision_time(f,repeats)
    CUDA.synchronize();t=time_ns()
    for _ in 1:repeats;f();end
    CUDA.synchronize()
    (time_ns()-t)/1e9/repeats
end

function run_precision_probe(out;batches=[1,8,32,99],samples=5,repeats=20)
    CUDA.functional() || error("CUDA functional device required")
    CUDA.allowscalar(false);mkpath(out)
    cfg=joinpath(@__DIR__,"..","config.yaml");rows=[]
    for batch in batches
        p,currents,cases,hu=_make_batch(cfg,batch,60,30)
        cpu=similar(hu);caches=[TDGLThermalCache(with_current(p,currents[k])) for k in 1:batch]
        cpu!()=begin
            for k in 1:batch;fdm_rhs_compatible!(view(cpu,:,k),view(hu,:,k),caches[k],0.0);end
        end
        u64=CuArray(hu);d64=similar(u64);c64=GPUFullCurlCache(p,currents)
        hu32=ComplexF32.(hu);u32=CuArray(hu32);d32=similar(u32);c32=GPUFullCurl32Cache(p,currents)
        f64!()=gpu_full_curl_rhs!(d64,u64,c64,0.0)
        f32!()=gpu_full_curl32_rhs!(d32,u32,c32,0.0)
        cpu!();f64!();f32!();CUDA.synchronize()
        e64=_precision_errors(Array(d64),cpu,p)
        e32=_precision_errors(ComplexF64.(Array(d32)),cpu,p)
        all(x -> x<=1e-11,values(e64)) || error("Float64 control mismatch")
        all(isfinite,values(e32)) || error("nonfinite lower-precision result")
        Array(u64)==hu && Array(u32)==hu32 || error("RHS changed input storage")
        timed=[]
        for rep in 1:samples
            vals=Dict{String,Float64}()
            order=isodd(rep) ? ("cpu","gpu64","gpu32") : ("gpu32","gpu64","cpu")
            for name in order
                f=name=="cpu" ? cpu! : name=="gpu64" ? f64! : f32!
                vals[name]=_precision_time(f,repeats)
            end
            push!(timed,(cpu_seconds=vals["cpu"],gpu64_seconds=vals["gpu64"],gpu32_seconds=vals["gpu32"],
                gpu64_over_gpu32=vals["gpu64"]/vals["gpu32"],cpu_over_gpu32=vals["cpu"]/vals["gpu32"]))
        end
        push!(rows,(batch=batch,initial_hashes=[c.initial_hash for c in cases],errors64=e64,errors32=e32,samples=timed,
            gpu64_over_gpu32_median=median(x.gpu64_over_gpu32 for x in timed),
            cpu_over_gpu32_median=median(x.cpu_over_gpu32 for x in timed)))
    end
    receipt=(scope="approximate-kernel-only",precision="ComplexF32 storage/Float32 coefficients with promoted intermediates in shared kernels",
        deterministic=true,trajectory_accuracy_claim=false,production_claim=false,created_utc=string(now(UTC)),rows=rows,
        julia_version=string(VERSION),device=string(CUDA.device()),cuda_runtime=string(CUDA.runtime_version()),
        benchmark_source_hash=bytes2hex(sha256(read(@__FILE__))),
        float32_source_hash=bytes2hex(sha256(read(joinpath(@__DIR__,"..","src","gpu_float32_experiment.jl")))),
        float64_source_hash=bytes2hex(sha256(read(joinpath(@__DIR__,"..","src","gpu_full_curl.jl")))),
        memory_limit_bytes=8_000_000_000)
    _write_json(joinpath(out,"gpu_float32_probe.json"),receipt)
    receipt
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_precision_probe(_arg("out","output/gpu_float32_probe"))
end
