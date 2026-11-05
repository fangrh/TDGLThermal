# Fixed-state cost isolation only. No solver trajectory or model change.
using JSON3, Random, Statistics, SHA, Dates

function probe_arg(name, default)
    i = findfirst(==("--" * name), ARGS)
    i === nothing ? default : ARGS[i + 1]
end

const PROBE_REPO = abspath(probe_arg("repo", joinpath(@__DIR__, "..")))
include(joinpath(PROBE_REPO, "bench", "fixtures.jl"))

function measured_calls(f, count)
    f()
    bytes = @allocated f()
    start = time_ns()
    for _ in 1:count
        f()
    end
    (; seconds_per_call=(time_ns() - start) / 1e9 / count, allocated_bytes=bytes)
end

function main()
    count = parse(Int, probe_arg("count", "5000"))
    repeats = parse(Int, probe_arg("repeats", "7"))
    out = abspath(probe_arg("out", "noise_cost_probe.json"))
    config = joinpath(PROBE_REPO, "studies", "full_curl_ndr", "config_T0.18.yaml")
    cases = []
    for prep in (:configured_down, :cool), noise in (0.0, 0.001)
        c = make_case(config; Je=0.18, noise=noise, seed=2026110411, preparation=prep)
        cache = TDGLThermalCache(c.params)
        u = copy(c.u0); du = similar(u)
        rhs = () -> fdm_rhs_compatible!(du, u, cache, 0.0)
        rhs()
        dpsi = () -> compute_dpsi_dt!(cache.dpsi, cache.RHS_psi, cache.state, c.params, cache.Tc)
        # Compilation/warmup is excluded from measurements.
        measured_calls(rhs, 10); measured_calls(dpsi, 10)
        push!(cases, (; preparation=string(prep), noise, initial_hash=c.initial_hash,
                       model_hash=c.model_hash, rhs_source_hash=c.rhs_source_hash,
                       config_hash=c.config_hash, rhs, dpsi))
    end
    for prep in ("configured_down", "cool")
        pair = filter(c -> c.preparation == prep, cases)
        pair[1].initial_hash == pair[2].initial_hash || error("noise cases have different input fields")
    end
    rows = []
    for rep in 1:repeats
        # Alternate order to reduce systematic warm/shared-load bias.
        for i in (isodd(rep) ? eachindex(cases) : reverse(eachindex(cases)))
            c = cases[i]
            Random.seed!(2026110411 + rep)
            rhs = measured_calls(c.rhs, count)
            dpsi = measured_calls(c.dpsi, count)
            push!(rows, (; repetition=rep, preparation=c.preparation, noise=c.noise,
                         initial_hash=c.initial_hash, rhs, dpsi))
        end
    end
    summary = []
    for prep in ("configured_down", "cool")
        med(noise, field) = median(getproperty(r, field).seconds_per_call for r in rows
                                  if r.preparation == prep && r.noise == noise)
        a, b = med(0.0, :rhs), med(0.001, :rhs)
        push!(summary, (; preparation=prep, noise_off_rhs_seconds=a, noise_on_rhs_seconds=b,
                         rhs_noise_on_over_off=b/a, rhs_increment_fraction=(b-a)/b,
                         noise_off_dpsi_seconds=med(0.0, :dpsi), noise_on_dpsi_seconds=med(0.001, :dpsi)))
    end
    provenance = [(; preparation=c.preparation, noise=c.noise, initial_hash=c.initial_hash,
                    model_hash=c.model_hash, rhs_source_hash=c.rhs_source_hash,
                    config_hash=c.config_hash) for c in cases]
    receipt = (; schema="tdgl-noise-cost-probe-v1", created_utc=string(now(UTC)),
               scope="Fixed-state RHS/dpsi microbenchmark; shared CPU load; excludes solve, output and GPU. Not a noise-on/off physical-equivalence test.",
               julia=string(VERSION), machine=Sys.MACHINE, grid=(60,30), Je=0.18,
               count, repeats, load_average=Sys.loadavg(), rows, summary, provenance,
               probe_sha256=bytes2hex(sha256(read(@__FILE__))))
    isfile(out) && error("refusing to overwrite $out")
    mkpath(dirname(out))
    open(out, "w") do io
        write(io, JSON3.write(receipt)); write(io, '\n')
    end
    println(JSON3.write(summary))
end

main()
