using Random, JSON3
length(ARGS)==2 || error("usage: model_path output_directory")
ENV["NOFASTSCAN_SKIP_MAIN"]="1"
ENV["NOFASTSCAN_SKIP_WORKERS"]="1"
ENV["TDGL_RHS_BACKEND"]="reference"
model=abspath(ARGS[1]); out=abspath(ARGS[2])
include(model)
include(joinpath(@__DIR__,"fixtures.jl"))
case=make_case(joinpath(dirname(model),"config.yaml");Je=0.18,noise=0.001,
               seed=2026110401,preparation=:configured_down)
p=case.params
state=initialize_state_clean(p)
vector_to_state!(state,copy(case.u0),p)
Random.seed!(case.seed)
mkpath(out)
result=run_stable_simulation_streaming(state,p,10.0,1.0,0.9,
    joinpath(out,"reference_check.h5"),p.Je,"down")
open(joinpath(out,"reference_check.json"),"w") do io
    write(io,JSON3.write((model,initial_hash=case.initial_hash,seed=case.seed,
        horizon=10.0,snapshot_interval=1.0,skip_ratio=0.9,noise=0.001,
        julia_version=string(VERSION),summary=result)))
end
println("Frozen reference comparison input hash: "*case.initial_hash)
