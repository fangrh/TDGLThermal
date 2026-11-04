#!/usr/bin/env julia
"""Matched per-point benchmark of the original four-window animation renderer.

Usage: julia benchmark_rendering.jl --reference REF.json --candidate CAND.json
       --out OUTDIR [--frames 20] [--fps 5]

The benchmark is intentionally separate from the CPU campaign: it measures
HDF loading plus the existing ``generate_single_Je_animation_remote`` render
for one matched point, not a full sweep.
"""
using Dates
using HDF5
using JSON3
using Printf
using SHA
using Statistics

function option(args, name, default=nothing)
    i = findfirst(==(name), args)
    i === nothing && return default
    i == length(args) && error("missing value for $name")
    return args[i + 1]
end

function required(args, name)
    value = option(args, name)
    value === nothing && error("required option $name")
    return value
end

function json_read(path::String)
    JSON3.read(read(path, String))
end

function sha256_file(path::String)
    bytes2hex(SHA.sha256(read(path)))
end

function manifest_data(path::String)
    x = json_read(path)
    isfile(path) || error("manifest not found: $path")
    x.complete == true || error("manifest incomplete: $path")
    x.status in ("success", "Success", "ReturnCode.Success") || error("manifest failed: $path")
    return x
end

function load_animation_data(path::String, Je::Float64, direction::String)
    h5open(path, "r") do file
        times = read(file, "times")
        V = read(file, "V")
        T = read(file, "T")
        psi = read(file, "psi_real") .+ im .* read(file, "psi_imag")
        Nx = size(psi, 1) - 1
        Ny = size(psi, 2) - 1
        T_avg = vec(mean(T[2:Nx, 2:Ny, :], dims=(1, 2)))
        V_t = [(times[i], V[i]) for i in eachindex(times)]
        return (Je=Je, direction=direction, psi=psi, T=T, times=times,
                skip_idx=1, T_avg=T_avg, V=V, V_avg=mean(V), V_t=V_t,
                Nx=Nx, Ny=Ny, hx=0.5, hy=0.5)
    end
end

function ffprobe_info(path::String)
    exe = Sys.which("ffprobe")
    exe === nothing && return Dict("available" => false, "reason" => "ffprobe not found")
    try
        raw = read(`$exe -v error -select_streams v:0 -show_entries stream=nb_frames,width,height -of json $path`, String)
        obj = JSON3.read(raw)
        isempty(obj.streams) && return Dict("available" => true, "error" => "no video stream")
        s = obj.streams[1]
        return Dict("available" => true, "frames" => s.nb_frames === nothing ? nothing : parse(Int, String(s.nb_frames)),
                    "width" => Int(s.width), "height" => Int(s.height))
    catch err
        return Dict("available" => true, "error" => sprint(showerror, err))
    end
end

function run_render(manifest_path::String, label::String, run_dir::String, frames::Int, fps::Int,
                    source_file::String, Je::Float64, solver_wall)
    manifest = manifest_data(manifest_path)
    h5_path = String(manifest.output_path)
    isfile(h5_path) || error("missing HDF5 output: $h5_path")
    mkpath(run_dir)
    # HDF loading is deliberately inside the measured interval.
    started = time_ns()
    data = load_animation_data(h5_path, Je, "up")
    Nx, Ny = data.Nx, data.Ny
    x = collect(range(-Nx * data.hx / 2 + data.hx, stop=Nx * data.hx / 2 - data.hx, length=Nx - 1))
    y = collect(range(data.hy, stop=Ny * data.hy - data.hy, length=Ny - 1))
    V_t_up = Dict{Float64, Vector{Tuple{Float64, Float64}}}(Je => data.V_t)
    V_t_down = Dict{Float64, Vector{Tuple{Float64, Float64}}}(Je => data.V_t)
    Je_points = [Je]; V_points = [data.V_avg]
    output = generate_single_Je_animation_remote(run_dir, Je, data, V_t_up, V_t_down, "up", x, y,
        maximum(data.T), max(maximum(data.V), 1e-6), Je, 1, Je_points, V_points,
        Je_points, V_points, Nx, Ny, fps, frames)
    elapsed = (time_ns() - started) / 1e9
    isfile(output) && filesize(output) > 0 || error("renderer did not create nonempty movie: $output")
    return Dict("backend" => label, "manifest_path" => manifest_path, "manifest_sha256" => sha256_file(manifest_path),
                "solver_HDF_wall_seconds" => Float64(manifest.wall_seconds),
                "render_wall_seconds" => elapsed,
                "measured_scope" => "point solve + HDF load + original animation render; not full-scan end-to-end",
                "frames_requested" => frames, "fps" => fps, "actual_movie_path" => output,
                "movie_size_bytes" => filesize(output), "ffprobe" => ffprobe_info(output),
                "hdf_output_path" => h5_path, "hdf_output_hash" => String(manifest.output_hash),
                "hdf_output_size_bytes" => filesize(h5_path), "skip_idx" => 1,
                "grid" => Dict("Nx" => Nx, "Ny" => Ny, "interior_x_points" => Nx - 1, "interior_y_points" => Ny - 1))
end

function warm_render(manifest_path::String, source_file::String, Je::Float64, frames::Int, fps::Int, outdir::String)
    # Compilation/warmup is excluded from all reported timings.
    warmdir = joinpath(outdir, "warmup")
    isdir(warmdir) && rm(warmdir; recursive=true, force=true)
    run_render(manifest_path, "warmup", warmdir, min(frames, 2), fps, source_file, Je, 0.0)
    rm(warmdir; recursive=true, force=true)
end

function main(args=ARGS)
    ref_path = required(args, "--reference")
    cand_path = required(args, "--candidate")
    outdir = required(args, "--out")
    frames = parse(Int, option(args, "--frames", "20"))
    fps = parse(Int, option(args, "--fps", "5"))
    source_file = option(args, "--source", joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl"))
    frames > 0 && fps > 0 || error("frames and fps must be positive")
    mkpath(outdir)
    # Main defines the exact production renderer.  Suppress its worker setup and main sweep.
    ENV["NOFASTSCAN_SKIP_WORKERS"] = "1"
    ENV["NOFASTSCAN_SKIP_MAIN"] = "1"
    include(source_file)
    ref = manifest_data(ref_path); cand = manifest_data(cand_path)
    Float64(ref.Je) == Float64(cand.Je) || error("reference/candidate Je mismatch")
    Je = Float64(ref.Je)
    warm_render(ref_path, source_file, Je, frames, fps, outdir)
    warm_render(cand_path, source_file, Je, frames, fps, outdir)
    runs = Dict[]
    # Alternate matched pairs to reduce order effects; three measured pairs.
    for repeat in 1:3
        for (label, path) in (("reference", ref_path), ("candidate", cand_path))
            run_dir = joinpath(outdir, "repeat$(repeat)_$(label)")
            push!(runs, merge(run_render(path, label, run_dir, frames, fps, source_file, Je,
                                         label == "reference" ? ref.wall_seconds : cand.wall_seconds),
                             Dict("repeat" => repeat)))
        end
    end
    result = Dict("schema" => "tdgl-rendering-benchmark-v1", "created_utc" => string(now(UTC)),
        "reference_manifest" => ref_path, "candidate_manifest" => cand_path,
        "reference_manifest_sha256" => sha256_file(ref_path), "candidate_manifest_sha256" => sha256_file(cand_path),
        "frames_requested" => frames, "fps" => fps, "data_skip_idx" => 1,
        "renderer" => "generate_single_Je_animation_remote from $(abspath(source_file))",
        "renderer_source_sha256" => sha256_file(source_file),
        "scope" => "measured original manifest solver HDF wall plus HDF load plus original animation render; not full-scan main end-to-end",
        "runs" => runs, "n_measured_pairs" => 3,
        "target_command" => "julia --project=. bench/benchmark_rendering.jl --reference REF.json --candidate CAND.json --out OUTDIR --frames 20 --fps 5")
    open(joinpath(outdir, "render_benchmark.json"), "w") do io
        write(io, JSON3.write(result)); write(io, '\n')
    end
    println(JSON3.write(Dict("output" => joinpath(outdir, "render_benchmark.json"), "n_measured_pairs" => 3)))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
