"""Checkpointed, ordered-current NDR history runner.

This is intentionally separate from reproduce_ndr.jl: each job owns an
ordered current history, while different jobs in one step remain independent.
"""

using Distributed
using JSON3, SHA, Dates, Printf, Serialization, HDF5
@everywhere using HDF5
include(joinpath(@__DIR__, "reproduce_ndr.jl"))

_hist_arg(name, default=nothing) = begin
    i = findfirst(==( "--" * name), ARGS)
    i === nothing ? default : (i == length(ARGS) ? error("missing value for --$name") : ARGS[i+1])
end
@everywhere _hist_hash(path) = isfile(path) ? bytes2hex(SHA.sha256(read(path))) : "missing"
_hist_field(x, k, default=nothing) = hasproperty(x, Symbol(k)) ? getproperty(x, Symbol(k)) : default
@everywhere _hist_required(x, k, label) = hasproperty(x, Symbol(k)) ? getproperty(x, Symbol(k)) : error("$label requires $k")
_hist_write(path, x) = open(path, "w") do io; write(io, JSON3.write(x)); write(io, '\n'); end

@everywhere function _hist_sidecar(h5::String)
    json = replace(h5, r"\.h5$" => ".json")
    isfile(json) || error("missing sidecar manifest for preceding HDF5: $json")
    json
end

@everywhere function _hist_load_fields(path, p)
    h5open(path, "r") do f
        psi_real = read(f["psi_real"]); psi_imag = read(f["psi_imag"])
        initialize_state_from_fields(p, complex.(psi_real[:, :, end], psi_imag[:, :, end]),
                                      read(f["Ax"])[:, :, end], read(f["Ay"])[:, :, end],
                                      read(f["T"])[:, :, end])
    end
end

@everywhere function _hist_run_step(job)
    cfg_hash = bytes2hex(SHA.sha256(read(job.config_path)))
    model_hash = bytes2hex(SHA.sha256(read(BENCH_MODEL_FILE)))
    previous_h5 = job.previous_h5
    source_h5 = previous_h5 === nothing ? job.initial_state_path : previous_h5
    previous_manifest_path = source_h5 === nothing ? nothing : _hist_sidecar(source_h5)
    if previous_manifest_path !== nothing
        pm = JSON3.read(read(previous_manifest_path))
        String(_hist_required(pm, "config_hash", "preceding manifest")) == cfg_hash ||
            error("config hash mismatch for $(job.id) step $(job.step)")
        String(_hist_required(pm, "model_hash", "preceding manifest")) == model_hash ||
            error("model hash mismatch for $(job.id) step $(job.step)")
        String(_hist_required(pm, "output_hash", "preceding manifest")) == _hist_hash(source_h5) ||
            error("preceding HDF5 hash mismatch for $(job.id) step $(job.step)")
    end
    c = make_case(job.config_path; Je=job.Je, noise=job.noise, seed=job.seed,
                  preparation=:configured_down, Nx=job.Nx, Ny=job.Ny,
                  initial_state_path=previous_h5 === nothing ? job.initial_state_path : nothing)
    if previous_h5 !== nothing
        state = _hist_load_fields(previous_h5, c.params)
        c = merge(c, (state=state, u0=state_to_vector(state), initial_hash=_state_hash(state),
                      initial_state_source=previous_h5, initial_source_hash=_hist_hash(previous_h5)))
    end
    rng_state = job.previous_rng === nothing ? nothing :
        open(job.previous_rng, "r") do io; deserialize(io); end
    rng_io = IOBuffer(); serialize(rng_io, rng_state === nothing ? (Random.seed!(job.seed); copy(Random.default_rng())) : rng_state)
    rng_before_hash = bytes2hex(SHA.sha256(take!(rng_io)))
    manifest = run_case(c, job.backend; horizon=job.horizon, tolerance=job.tolerance,
                        output_dir=job.output_dir, save_fields=true,
                        snapshot_interval=job.snapshot_interval, skip_ratio=job.skip_ratio,
                        rng_state=rng_state)
    rng_after = copy(Random.default_rng()); rng_after_path = joinpath(job.output_dir, "rng_after.bin")
    open(rng_after_path, "w") do io; serialize(io, rng_after); end
    (manifest=manifest, manifest_path=joinpath(job.output_dir,
        "$(job.backend)_Je$(job.Je)_seed$(job.seed).json"), rng_after_path=rng_after_path,
     rng_before_hash=rng_before_hash, rng_after_hash=_hist_hash(rng_after_path))
end

function hist_main()
    nprocs() > 1 || error("start Julia with -p N (N >= 1)")
    jobs_path = _hist_arg("jobs"); jobs_path === nothing && error("--jobs JSON path is required")
    out_arg = _hist_arg("out"); out_arg === nothing && error("--out is required")
    out = abspath(out_arg); isfile(jobs_path) || error("jobs file not found: $jobs_path")
    horizon = parse(Float64, _hist_arg("horizon", "5000")); horizon > 0 || error("horizon must be positive")
    skip = parse(Float64, _hist_arg("skip", "0.6")); 0 <= skip <= 1 || error("skip must be in [0,1]")
    tolerance = parse(Float64, _hist_arg("tolerance", "1e-4")); tolerance > 0 || error("tolerance must be positive")
    backend = String(_hist_arg("backend", "cpu_compatible")); backend in ("reference", "cpu_compatible") || error("unsupported backend")
    raw = JSON3.read(read(jobs_path)); raw isa JSON3.Array || error("jobs must be a JSON array")
    jobs = NamedTuple[]; ids = Set{String}()
    for item in raw
        id = String(_hist_required(item, "id", "each job")); id in ids && error("duplicate id $id"); push!(ids, id)
        cfg = abspath(String(_hist_required(item, "config_path", "job $id")))
        currents = Float64.(collect(_hist_required(item, "currents", "job $id"))); isempty(currents) && error("empty currents for $id")
        initial = _hist_field(item, "initial_state_path", nothing); initial = initial === nothing ? nothing : abspath(String(initial))
        initial_rng = _hist_field(item, "initial_rng_path", nothing); initial_rng = initial_rng === nothing ? nothing : abspath(String(initial_rng))
        initial_rng === nothing && error("job $id requires initial_rng_path")
        initial === nothing && error("job $id requires initial_state_path")
        isfile(initial) || error("missing initial state $initial"); initial_rng !== nothing && !isfile(initial_rng) && error("missing initial RNG $initial_rng")
        seed = Int(_hist_required(item, "seed", "job $id")); Nx = Int(_hist_field(item, "Nx", 60)); Ny = Int(_hist_field(item, "Ny", 30))
        noise = Float64(_hist_field(item, "noise", get_param(load_config_yaml(cfg), "noise", "strength", 0.001)))
        prior = Float64(_hist_field(item, "prior_cumulative_horizon", 50000.0))
        push!(jobs, (id=id, config_path=cfg, currents=currents, seed=seed, initial_state_path=initial,
                     prior_cumulative_horizon=prior,
                     initial_rng=initial_rng, Nx=Nx, Ny=Ny, noise=noise))
    end
    mkpath(out); warm = first(jobs)
    infos = [remotecall_fetch(_ws_worker_info, w, warm.config_path, _WS_CPU_FILE, joinpath(@__DIR__, "fixtures.jl"), _WS_MODEL_FILE, joinpath(@__DIR__, "..", "src", "cpu_compatible.jl")) for w in workers()]
    fetch.([remotecall(_ws_warm_worker, w, warm.config_path, joinpath(out, "warmup"), warm.noise, :configured_down, warm.Nx, warm.Ny, warm.seed, 2.0, tolerance, skip, 1.0) for w in workers()])
    previous = Dict(j.id => j.initial_state_path for j in jobs); previous_rng = Dict(j.id => j.initial_rng for j in jobs); chains = Dict(j.id => Any[] for j in jobs)
    maxsteps = maximum(length(j.currents) for j in jobs)
    for step in 1:maxsteps
        specs = NamedTuple[]
        for j in jobs
            step <= length(j.currents) || continue
            segdir = joinpath(out, j.id, "step$(step)_Je$(j.currents[step])"); isfile(joinpath(segdir, "chain.json")) && error("refusing to overwrite $segdir")
            push!(specs, (id=j.id, step=step, config_path=j.config_path, Je=j.currents[step], noise=j.noise, seed=j.seed,
                Nx=j.Nx, Ny=j.Ny, initial_state_path=step == 1 ? j.initial_state_path : nothing,
                previous_h5=step == 1 ? nothing : previous[j.id], previous_rng=step == 1 ? j.initial_rng : previous_rng[j.id],
                prior_cumulative_horizon=j.prior_cumulative_horizon,
                backend=backend, horizon=horizon, tolerance=tolerance, snapshot_interval=1.0, skip_ratio=skip, output_dir=segdir))
        end
        results = pmap(_hist_run_step, specs)
        for (spec, result) in zip(specs, results)
            r=result.manifest; entry=(id=spec.id, step=spec.step, Je=spec.Je, seed=spec.seed,
                cumulative_start=spec.prior_cumulative_horizon+(spec.step-1)*horizon, cumulative_end=spec.prior_cumulative_horizon+spec.step*horizon,
                current_change=spec.step == 1 ? "initial" : "abrupt", skip_ratio=skip, source_initial_horizon=spec.prior_cumulative_horizon,
                preceding_hdf5=spec.step == 1 ? spec.initial_state_path : spec.previous_h5,
                preceding_rng=spec.previous_rng, manifest=r, manifest_path=result.manifest_path,
                rng_before_sha256=result.rng_before_hash, rng_after_sha256=result.rng_after_hash, rng_after_path=result.rng_after_path)
            push!(chains[spec.id], entry); _hist_write(joinpath(spec.output_dir, "chain.json"), entry); previous[spec.id]=r.output_path; previous_rng[spec.id]=result.rng_after_path
        end
    end
    campaign=(schema_version="tdgl-ndr-history-v1", created_utc=string(now(UTC)), backend=backend, horizon=horizon, skip_ratio=skip, tolerance=tolerance, workers=workers(), nprocs=nprocs(), jobs_path=abspath(jobs_path), worker_info=infos, chains=chains, protocol="abrupt current steps; dwell and skip ratio recorded above; preparation horizon recorded per chain", scope_note="ordered history; not exact historical-paper claim")
    _hist_write(joinpath(out, "campaign.json"), campaign)
end

if abspath(PROGRAM_FILE) == @__FILE__
    hist_main()
end
