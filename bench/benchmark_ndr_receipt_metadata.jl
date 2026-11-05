#!/usr/bin/env julia
# Benchmark predecessor receipt-hash metadata only.
# The cached expected hash is the previous segment's already-verified output
# hash; the child hash is the fresh initial-state hash recorded after loading it.
# This measures only the receipt-hash component, not concurrent simulation load.
using JSON3, SHA, Statistics, Dates
include(joinpath(@__DIR__, "ndr_receipt_hashes.jl"))

_arg(name, default=nothing) = begin
    i = findfirst(==("--" * name), ARGS)
    i === nothing ? default : (i == length(ARGS) ? error("missing value for --$name") : ARGS[i+1])
end
_sha(path) = bytes2hex(SHA.sha256(read(path)))
_j(path) = JSON3.read(read(path))
_req(x, key, label) = hasproperty(x, Symbol(key)) ? getproperty(x, Symbol(key)) : error("$label requires $key")

function main()
    if "--help" in ARGS
        println("usage: julia benchmark_ndr_receipt_metadata.jl --campaign-root DIR --segment N --repeats N --out FILE")
        return
    end
    rootarg = _arg("campaign-root"); rootarg === nothing && error("--campaign-root is required")
    seg = parse(Int, _arg("segment", "2")); seg >= 2 || error("--segment must be at least 2")
    repeats = parse(Int, _arg("repeats", "3")); repeats > 0 || error("--repeats must be positive")
    outarg = _arg("out"); outarg === nothing && error("--out is required")
    out = abspath(outarg); isfile(out) && error("refusing to overwrite existing output: $out")
    root = abspath(rootarg)
    paths = String[]
    for (dir, _, files) in walkdir(root)
        target = joinpath(dir, "chain.json")
        endswith(dir, "segment$(seg)") && isfile(target) && push!(paths, target)
    end
    sort!(paths)
    isempty(paths) && error("no segment$(seg) chain receipts under $root")
    records = NamedTuple[]
    for child_path in paths
        child = _j(child_path)
        _req(child, "id", child_path)
        _req(child, "preceding_hdf5", child_path)
        _req(child, "preceding_sha256", child_path)
        prior_path = joinpath(dirname(dirname(child_path)), "segment$(seg-1)", "chain.json")
        seg >= 2 || error("segment 1 has no predecessor for this benchmark")
        isfile(prior_path) || error("missing predecessor receipt: $prior_path")
        prior = _j(prior_path)
        output_hash = _req(_req(prior, "manifest", prior_path), "output_hash", prior_path)
        preceding = abspath(String(child.preceding_hdf5)); isfile(preceding) || error("missing predecessor HDF5: $preceding")
        source_hash = _req(_req(child, "manifest", child_path), "initial_source_hash", child_path)
        expected = String(output_hash)
        checked_preceding_sha(preceding, source_hash, expected) == String(child.preceding_sha256) || error("receipt hash mismatch: $child_path")
        push!(records, (child_path=child_path, prior_path=prior_path, preceding=preceding,
                        source_hash=String(source_hash), expected=expected))
    end
    # Warm the first file and helper calls outside the measured regions.
    _sha(records[1].preceding)
    for r in records; checked_preceding_sha(r.preceding, r.source_hash, r.expected); end
    old_seconds = Float64[]; new_seconds = Float64[]
    for trial in 1:repeats
        GC.gc()
        push!(old_seconds, @elapsed begin
            for r in records
                got = bytes2hex(SHA.sha256(read(r.preceding)))
                got == r.expected || error("old hash mismatch")
            end
        end)
        GC.gc()
        push!(new_seconds, @elapsed begin
            for r in records
                got = checked_preceding_sha(r.preceding, r.source_hash, r.expected)
                got == r.expected || error("helper mismatch")
            end
        end)
    end
    old_med, new_med = median(old_seconds), median(new_seconds)
    result = (schema="tdgl-ndr-receipt-metadata-benchmark-v1", scope="parent receipt hash component only; concurrent simulation load; no whole-campaign speedup claimed",
              campaign_root=root, segment=seg, repeats=repeats, point_count=length(records),
              total_preceding_bytes=sum(filesize(r.preceding) for r in records), old_seconds=old_seconds,
              new_seconds=new_seconds, old_median_seconds=old_med, new_median_seconds=new_med,
              new_over_old_component_cost=old_med > 0 ? new_med / old_med : nothing,
              helper_sha256=_sha(joinpath(@__DIR__, "ndr_receipt_hashes.jl")),
              benchmark_sha256=_sha(@__FILE__), parent_chain_paths=[r.prior_path for r in records],
              parent_chain_sha256=[_sha(r.prior_path) for r in records],
              child_chain_sha256=[_sha(r.child_path) for r in records],
              parent_output_hashes=[r.expected for r in records], julia_version=string(VERSION),
              machine=Sys.MACHINE, created_utc=string(now(UTC)))
    open(out, "w") do io; write(io, JSON3.write(result)); write(io, '\n'); end
    println("Wrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__; main(); end
