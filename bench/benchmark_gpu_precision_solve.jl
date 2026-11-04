# Paired deterministic F64 versus ComplexF32-storage GPU solver experiment.
# This is an accuracy-gated approximation experiment, not a production claim.
using CUDA
using OrdinaryDiffEq
using JSON3
using SHA
using Statistics
using Dates
using HDF5

const ROOT=normpath(joinpath(@__DIR__,".."))
const BENCH=joinpath(ROOT,"bench","benchmark_gpu_solve.jl")
const F32SRC=joinpath(ROOT,"src","gpu_float32_experiment.jl")
include(BENCH)
include(F32SRC)

_arg(n,d=nothing)=(i=findfirst(==("--"*n),ARGS);i===nothing ? d : (i==length(ARGS) ? error("missing value for --$n") : ARGS[i+1]))
function _json(p,x); open(p,"w") do io; write(io,JSON3.write(x)); write(io,'\n'); end end
_hash(p)=bytes2hex(SHA.sha256(read(p)))

function _f32_solve(p,currents,h_u,times,tol;warmup=false)
    d_u=CuArray(ComplexF32.(h_u)); cache=GPUFullCurl32Cache(p,currents)
    rhs! = (du,u,_,t)->gpu_full_curl32_rhs!(du,u,cache,t)
    prob=ODEProblem(rhs!,d_u,(times[1],times[end]),nothing)
    kw=(;abstol=tol,reltol=tol,saveat=times,tstops=times,save_everystep=false,dense=false,internalnorm=_batch_internalnorm)
    warmup && (solve(prob,Tsit5();kw...);CUDA.synchronize())
    CUDA.synchronize();t0=time_ns();sol=solve(prob,Tsit5();kw...);CUDA.synchronize()
    seconds=(time_ns()-t0)/1e9
    _assert_solution(sol,times)
    (solution=sol,seconds=seconds,stats=_stats(sol),setup_bytes=sizeof(ComplexF32)*length(h_u))
end

function _load_reference(root,case,times)
    path=joinpath(root,"seed$(case.seed)_reference_states.h5"); isfile(path)||error("missing $path")
    ref=h5open(path,"r") do f
        all(isapprox.(read(f["times"]),times;rtol=0,atol=1e-10)) || error("common times mismatch for seed $(case.seed)")
        re=read(f["real"]); im=read(f["imag"]); [complex.(re[:,k],im[:,k]) for k in axes(re,2)]
    end
    checkpath=joinpath(root,"seed$(case.seed)_check.json"); isfile(checkpath)||error("missing $checkpath")
    check=JSON3.read(read(checkpath,String));digest=_hash(path)
    String(check.initial_hash)==case.initial_hash || error("initial state provenance mismatch")
    String(check.reference_states_hash)==digest || error("reference artifact hash mismatch")
    (states=ref,check=check,file_hash=digest)
end

function _budget(check,f)
    Float64(check["allowed_budget"][String(f)])
end
function _converged(check); Bool(check["reference_converged"]) end

function run_precision(;out,reference_root,config,batch=99,horizon=100.0,tol=1e-6,repetitions=3)
    batch==99 || error("this gate is defined for --batch 99 only")
    horizon>0 && tol>0 && repetitions>0 || error("invalid benchmark setting")
    CUDA.functional() || error("CUDA.functional() is false"); CUDA.allowscalar(false); mkpath(out)
    p,currents,cases=_cases(config,batch,60,30,:cool); times=collect(0.0:1.0:horizon)
    times[end]==horizon || push!(times,horizon)
    estimated_bytes=3*length(cases[1].u0)*batch*(length(times)+54)*(sizeof(ComplexF64)+sizeof(ComplexF32))
    estimated_bytes<8*1024^3 || error("estimated live device memory exceeds 8 GiB")
    h_u=Matrix{ComplexF64}(undef,length(cases[1].u0),batch); for b in 1:batch;h_u[:,b].=cases[b].u0;end
    warmtimes=[0.0,min(2.0,horizon)]
    _gpu_solve(p,currents,h_u,tol,warmtimes); _f32_solve(p,currents,h_u,warmtimes,tol)
    s64=Any[];s32=Any[];last64=nothing;last32=nothing
    for rep in 1:repetitions
        if isodd(rep)
            a=_gpu_solve(p,currents,h_u,tol,times); b=_f32_solve(p,currents,h_u,times,tol)
        else
            b=_f32_solve(p,currents,h_u,times,tol); a=_gpu_solve(p,currents,h_u,tol,times)
        end
        push!(s64,(seconds=a.seconds,stats=a.stats));push!(s32,(seconds=b.seconds,stats=b.stats));last64=a;last32=b
    end
    st64=[Array(x) for x in last64.solution.u]; st32=[ComplexF64.(Array(x)) for x in last32.solution.u]
    rows=Any[];fail64=Int[];fail32=Int[];conv=0
    for j in 1:batch
        art=_load_reference(reference_root,cases[j],times); ref=art.states; chk=art.check
        pj=with_current(p,cases[j].Je)
        _converged(chk)&&(conv+=1); obsref=_observables(ref,pj)
        e64=_all_errors([st64[k][:,j] for k in eachindex(times)],ref,pj,_observables([st64[k][:,j] for k in eachindex(times)],pj),obsref)
        e32=_all_errors([st32[k][:,j] for k in eachindex(times)],ref,pj,_observables([st32[k][:,j] for k in eachindex(times)],pj),obsref)
        good64=_converged(chk)&&all(e64[f]<=_budget(chk,f) for f in keys(e64));good32=_converged(chk)&&all(e32[f]<=_budget(chk,f) for f in keys(e32))
        c64=[_state_checks(st64[k][:,j],p) for k in eachindex(times)];c32=[_state_checks(st32[k][:,j],p) for k in eachindex(times)]
        good64 &= all(x -> x.finite && x.max_imag_A<=1e-12 && x.max_imag_T<=1e-12,c64)
        good32 &= all(x -> x.finite && x.max_imag_A<=1e-12 && x.max_imag_T<=1e-12,c32)
        good64||push!(fail64,j);good32||push!(fail32,j)
        push!(rows,(batch_index=j,seed=cases[j].seed,current=cases[j].Je,reference_file_hash=art.file_hash,reference_converged=_converged(chk),
            f64_errors=e64,f32_errors=e32,f64_accurate=good64,f32_accurate=good32,
            f64_checks=(finite=all(x->x.finite,c64),max_imag_A=maximum(x->x.max_imag_A,c64),max_imag_T=maximum(x->x.max_imag_T,c64),
                min_temperature=minimum(x.min_temperature for x in c64),max_temperature=maximum(x.max_temperature for x in c64)),
            f32_checks=(finite=all(x->x.finite,c32),max_imag_A=maximum(x->x.max_imag_A,c32),max_imag_T=maximum(x->x.max_imag_T,c32),
                min_temperature=minimum(x.min_temperature for x in c32),max_temperature=maximum(x.max_temperature for x in c32))))
    end
    manifest=(schema_version="tdgl-gpu-precision-solver-v1",scope="deterministic-f64-vs-f32-storage-whole-solver",
        approximation=true,production_noise_claim=false,full_output_claim=false,batch=batch,horizon=horizon,tolerance=tol,repetitions=repetitions,
        precisions=(control="ComplexF64/Float64",candidate="ComplexF32 storage/Float32 coefficients with promoted intermediates"),
        reference_root=abspath(reference_root),config_hash=_hash(config),gpu_source_hash=_hash(joinpath(ROOT,"src","gpu_full_curl.jl")),
        f32_source_hash=_hash(F32SRC),model_hash=_hash(joinpath(ROOT,"current_sweep_thermal_nofastscan.jl")),julia_version=string(VERSION),created_utc=string(now(UTC)),
        reference_converged_count=conv,f64_accurate_count=batch-length(fail64),f32_accurate_count=batch-length(fail32),
        f64_failures=fail64,f32_failures=fail32,all99_f64_pass=isempty(fail64),all99_f32_pass=isempty(fail32),
        f64_samples=s64,f32_samples=s32,f64_median=median(getproperty.(s64,:seconds)),f32_median=median(getproperty.(s32,:seconds)),
        memory_limit_bytes=8*1024^3,estimated_bytes=estimated_bytes,benchmark_source_hash=_hash(@__FILE__),rows=rows)
    _json(joinpath(out,"gpu_precision_solver.json"),manifest);manifest
end

if abspath(PROGRAM_FILE)==@__FILE__
    out=_arg("out");out===nothing&&error("--out is required");ref=_arg("reference-root");ref===nothing&&error("--reference-root is required")
    result=run_precision(out=String(out),reference_root=String(ref),config=String(_arg("config",joinpath(ROOT,"config.yaml"))),
        batch=parse(Int,_arg("batch","99")),horizon=parse(Float64,_arg("horizon","100")),tol=parse(Float64,_arg("tolerance","1e-6")),repetitions=parse(Int,_arg("repetitions","3")))
    result.all99_f32_pass || exit(2)
end
