"""Checkpointed full-curl NDR continuation campaign.

Run with ``julia -p N --project=. bench/reproduce_ndr.jl ...``.  Jobs in one
segment are independent and are dispatched with persistent workers; segments
are deliberately serialized so each state chain remains explicit.
"""

using Distributed
using JSON3, SHA, Dates, Printf, Serialization
@everywhere using Serialization

const _NDR_WORKER_SCRIPT = joinpath(@__DIR__, "benchmark_worker_scaling.jl")
include(_NDR_WORKER_SCRIPT) # loads the model, helpers, and persistent workers

_ndr_arg(name, default=nothing) = begin
    i = findfirst(==("--" * name), ARGS)
    i === nothing ? default : (i == length(ARGS) ? error("missing value for --$name") : ARGS[i+1])
end

_hash(path) = isfile(path) ? bytes2hex(SHA.sha256(read(path))) : "missing"
_field(x, k, default=nothing) = hasproperty(x, Symbol(k)) ? getproperty(x, Symbol(k)) : default
_required(x, k, label) = hasproperty(x, Symbol(k)) ? getproperty(x, Symbol(k)) : error("$label requires $k")

function _write_json(path, value)
    open(path, "w") do io
        write(io, JSON3.write(value)); write(io, '\n')
    end
end

@everywhere function _ndr_run_segment(job)
    # make_case enforces Je equality when loading a prior HDF5 state and resets
    # the legacy RNG inside run_case, preserving the production draw policy.
    c = make_case(job.config_path; Je=job.Je, noise=job.noise, seed=job.segment_seed,
                  preparation=:configured_down, Nx=job.Nx, Ny=job.Ny,
                  initial_state_path=job.initial_state_path)
    rng_state = job.rng_state_path === nothing ? nothing :
        open(job.rng_state_path, "r") do io; deserialize(io); end
    rng_before = if rng_state === nothing
        Random.seed!(job.segment_seed)
        copy(Random.default_rng())
    else
        rng_state
    end
    rng_io = IOBuffer(); serialize(rng_io, rng_before)
    rng_before_hash = bytes2hex(SHA.sha256(take!(rng_io)))
    manifest = run_case(c, job.backend; horizon=job.horizon, tolerance=job.tolerance,
             output_dir=job.output_dir, save_fields=true,
             snapshot_interval=job.snapshot_interval, skip_ratio=job.skip_ratio,
             rng_state=rng_state)
    rng_after_path = joinpath(job.output_dir, "rng_after.bin")
    rng_after = copy(Random.default_rng())
    open(rng_after_path, "w") do io; serialize(io, rng_after); end
    (manifest=manifest, rng_after_path=rng_after_path,
     rng_before_hash=rng_before_hash, rng_after_hash=bytes2hex(SHA.sha256(read(rng_after_path))))
end

function ndr_main()
    nprocs() > 1 || error("start Julia with -p N (N >= 1)")
    jobs_path = _ndr_arg("jobs"); jobs_path === nothing && error("--jobs JSON path is required")
    out_arg = _ndr_arg("out"); out_arg === nothing && error("--out is required")
    out = abspath(out_arg)
    isfile(jobs_path) || error("jobs file not found: $jobs_path")
    backend = String(_ndr_arg("backend", "cpu_compatible"))
    backend in ("reference", "cpu_compatible") || error("unsupported backend: $backend")
    segments = parse(Int, _ndr_arg("segments", "1")); segments > 0 || error("segments must be positive")
    horizon = parse(Float64, _ndr_arg("horizon", "5000")); horizon > 0 || error("horizon must be positive")
    skip = parse(Float64, _ndr_arg("skip", "0.6")); 0 <= skip <= 1 || error("skip must be in [0,1]")
    snapshot = parse(Float64, _ndr_arg("snapshot-interval", "1")); snapshot > 0 || error("snapshot interval must be positive")
    tolerance = parse(Float64, _ndr_arg("tolerance", "1e-4")); tolerance > 0 || error("tolerance must be positive")

    raw = JSON3.read(read(jobs_path))
    raw isa JSON3.Array || error("--jobs must contain a JSON array")
    isempty(raw) && error("jobs array is empty")
    jobs = NamedTuple[]
    ids = Set{String}()
    for item in raw
        id = String(_required(item, "id", "each job"))
        id in ids && error("duplicate job id: $id"); push!(ids, id)
        cfg = abspath(String(_required(item, "config_path", "job $id")))
        Je = Float64(_required(item, "Je", "job $id"))
        seed = Int(_required(item, "seed", "job $id"))
        Nx = Int(_field(item, "Nx", 60)); Ny = Int(_field(item, "Ny", 30))
        initial = _field(item, "initial_state_path", nothing)
        initial_path = initial === nothing ? nothing : abspath(String(initial))
        initial_rng = _field(item, "initial_rng_path", nothing)
        initial_rng_path = initial_rng === nothing ? nothing : abspath(String(initial_rng))
        prior = Float64(_field(item, "prior_cumulative_horizon", 0.0))
        initial_path !== nothing && !isfile(initial_path) && error("missing initial state for $id: $initial_path")
        initial_rng_path !== nothing && !isfile(initial_rng_path) && error("missing initial RNG state for $id: $initial_rng_path")
        push!(jobs, (id=id, config_path=cfg, Je=Je, seed=seed, Nx=Nx, Ny=Ny,
                     noise=Float64(_field(item, "noise", NaN)), initial_state_path=initial_path,
                     initial_rng_path=initial_rng_path, prior_cumulative_horizon=prior))
    end
    # NaN means use the configured noise strength; this keeps jobs concise while
    # retaining an explicit override for controlled campaigns.
    jobs = [merge(j, (noise=isnan(j.noise) ? Float64(get_param(load_config_yaml(j.config_path), "noise", "strength", 0.001)) : j.noise,)) for j in jobs]

    mkpath(out)
    warm_cfg = first(jobs).config_path
    infos = [remotecall_fetch(_ws_worker_info, w, warm_cfg, _WS_CPU_FILE,
                              joinpath(@__DIR__, "fixtures.jl"), _WS_MODEL_FILE,
                              joinpath(@__DIR__, "..", "src", "cpu_compatible.jl")) for w in workers()]
    warmups = [remotecall(_ws_warm_worker, w, warm_cfg, joinpath(out, "warmup"),
                          first(jobs).noise, :configured_down, first(jobs).Nx, first(jobs).Ny,
                          first(jobs).seed, 2.0, tolerance, skip, snapshot) for w in workers()]
    fetch.(warmups)

    chains = Dict{String,Vector{Any}}(j.id => Any[] for j in jobs)
    previous = Dict{String,Union{Nothing,String}}(j.id => j.initial_state_path for j in jobs)
    previous_rng = Dict{String,Union{Nothing,String}}(j.id => j.initial_rng_path for j in jobs)
    cumulative = Dict(j.id => j.prior_cumulative_horizon for j in jobs)
    for segment in 1:segments
        runjobs = NamedTuple[]
        for j in jobs
            segdir = joinpath(out, j.id, "segment$(segment)")
            isfile(joinpath(segdir, "chain.json")) && error("refusing to overwrite $segdir")
            push!(runjobs, (config_path=j.config_path, Je=j.Je, noise=j.noise,
                segment_seed=j.seed, Nx=j.Nx, Ny=j.Ny,
                initial_state_path=previous[j.id], backend=backend, horizon=horizon,
                tolerance=tolerance, snapshot_interval=snapshot, skip_ratio=skip,
                rng_state_path=previous_rng[j.id],
                output_dir=segdir))
        end
        manifests = pmap(_ndr_run_segment, runjobs)
        for (j, result, spec) in zip(jobs, manifests, runjobs)
            r = result.manifest
            start = cumulative[j.id]; stop = start + horizon
            preceding = previous[j.id]
            entry = (id=j.id, Je=j.Je, seed=j.seed, segment_seed=spec.segment_seed,
                segment=segment, cumulative_start=start, cumulative_end=stop,
                preceding_hdf5=preceding, preceding_sha256=preceding === nothing ? nothing : _hash(preceding),
                manifest=r, full_field_output=true, backend=backend,
                rng_before_sha256=result.rng_before_hash, rng_after_sha256=result.rng_after_hash,
                rng_after_path=result.rng_after_path, preceding_rng_path=spec.rng_state_path,
                rng_protocol=spec.rng_state_path === nothing ? "seed_case_first_segment" : "restore_serialized_checkpoint")
            push!(chains[j.id], entry); _write_json(joinpath(spec.output_dir, "chain.json"), entry)
            previous[j.id] = r.output_path; cumulative[j.id] = stop
            previous_rng[j.id] = result.rng_after_path
        end
    end
    source_paths = [(@__FILE__), _NDR_WORKER_SCRIPT, _WS_CPU_FILE, _WS_MODEL_FILE, joinpath(@__DIR__, "..", "src", "cpu_compatible.jl")]
    campaign = (schema_version="tdgl-ndr-campaign-v1", created_utc=string(now(UTC)), backend=backend,
        segments=segments, horizon=horizon, skip_ratio=skip, snapshot_interval=snapshot,
        tolerance=tolerance, workers=workers(), nprocs=nprocs(), jobs_path=abspath(jobs_path), jobs_sha256=_hash(jobs_path),
        source_hashes=Dict(basename(p) => _hash(p) for p in source_paths), worker_info=infos,
        chains=chains, scope_note="full-curl physical campaign; not a claim of exact historical-paper reproduction")
    _write_json(joinpath(out, "campaign.json"), campaign)
    println("Wrote ", joinpath(out, "campaign.json"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    ndr_main()
end
