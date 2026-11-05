# All-column achieved-accuracy validation. Start with Julia -p N; this is
# separate from controlled timing and configured-noise production runs.
using Distributed,JSON3,SHA,Dates,HDF5
include(joinpath(@__DIR__,"benchmark_gpu_solve.jl"))
nprocs()>1 || error("start this validator with julia -p N")
const _VALIDATION_BENCH=joinpath(@__DIR__,"benchmark_gpu_solve.jl")
@sync for w in workers()
    @async remotecall_fetch(Base.include,w,Main,_VALIDATION_BENCH)
end

@everywhere function _validation_save_states(path,states,times)
    values=hcat(states...)
    h5open(path,"w") do f
        f["times"]=times;f["real"]=real.(values);f["imag"]=imag.(values)
    end
end

@everywhere function _validation_job(job)
    case,gs,index,times,out=job
    start=time_ns()
    result=_reference_check(case,nothing,gs,times;retain_states=true)
    c=result.check
    prefix=joinpath(out,"seed$(case.seed)")
    refpath=prefix*"_reference_states.h5";basepath=prefix*"_baseline_states.h5"
    gpupath=prefix*"_gpu_states.h5"
    _validation_save_states(refpath,result.reference_states,times)
    _validation_save_states(basepath,result.baseline_states,times)
    _validation_save_states(gpupath,gs,times)
    receipt=merge(c,(batch_index=index,initial_hash=case.initial_hash,
        reference_states_path=refpath,baseline_states_path=basepath,gpu_states_path=gpupath,
        reference_states_hash=bytes2hex(SHA.sha256(read(refpath))),
        baseline_states_hash=bytes2hex(SHA.sha256(read(basepath))),
        gpu_states_hash=bytes2hex(SHA.sha256(read(gpupath))),
        elapsed_seconds=(time_ns()-start)/1e9))
    open(prefix*"_check.json","w") do io;write(io,JSON3.write(receipt));end
    receipt
end

function validate_batch(out,config;batch=99,horizon=100.0)
    mkpath(out);times=collect(0.0:1.0:horizon)
    times[end]==horizon || push!(times,horizon)
    p,currents,cases=_cases(config,batch,60,30,:cool)
    hu=hcat([c.u0 for c in cases]...)
    _gpu_solve(p,currents,hu,1e-6,[0.0,min(2.0,horizon)])
    gpu=_gpu_solve(p,currents,hu,1e-6,times)
    CUDA.synchronize();states=[Array(x) for x in gpu.solution.u]
    jobs=[(cases[b],[states[k][:,b] for k in eachindex(times)],b,times,out) for b in 1:batch]
    results=pmap(_validation_job,jobs)
    failures=[r.batch_index for r in results if !r.accurate]
    data=(schema_version="tdgl-gpu-batch-accuracy-v2",scope="all-column deterministic achieved accuracy, not production-noise validation",
        batch_count=batch,horizon=horizon,tolerance=1e-6,workers=nworkers(),
        accurate_count=count(r -> r.accurate,results),converged_count=count(r -> r.reference_converged,results),
        failure_ids=failures,all_accuracy_pass=isempty(failures),
        config_hash=bytes2hex(sha256(read(config))),model_hash=bytes2hex(sha256(read(BENCH_MODEL_FILE))),
        gpu_source_hash=bytes2hex(sha256(read(joinpath(@__DIR__,"..","src","gpu_full_curl.jl")))),
        benchmark_source_hash=bytes2hex(sha256(read(@__FILE__))),
        reference_helper_hash=bytes2hex(sha256(read(_VALIDATION_BENCH))),
        julia_version=string(VERSION),created_utc=string(now(UTC)),gpu_seconds=gpu.seconds,results=results)
    _write_json(joinpath(out,"summary.json"),data)
    println(JSON3.write((batch_count=batch,accurate_count=data.accurate_count,converged_count=data.converged_count,failure_ids=failures)))
    data
end

if abspath(PROGRAM_FILE)==@__FILE__
    result=validate_batch(String(_arg("out")),String(_arg("config",joinpath(@__DIR__,"..","config.yaml")));
        batch=parse(Int,_arg("batch","99")),horizon=parse(Float64,_arg("horizon","100")))
    result.all_accuracy_pass || exit(2)
end
