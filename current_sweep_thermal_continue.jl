using Dates
using Distributed
using Printf

ENV["NOFASTSCAN_SKIP_MAIN"] = "1"
include(joinpath(@__DIR__, "current_sweep_thermal_nofastscan.jl"))

@everywhere struct ContinuationPoint
    Je::Float64
    direction::String
    file::String
end

Base.@kwdef struct ContinuationOptions
    config_path::String
    input_folder::String
    start_current::Float64
    end_current::Float64
    stable_time::Float64
    ramp_time::Float64
    dry_run::Bool = false
end

function discover_continuation_points(input_folder::String, direction::String)
    isdir(input_folder) || error("input_folder does not exist or is not a directory: $input_folder")
    direction in ("up", "down") || error("direction must be 'up' or 'down', got: $direction")

    points = ContinuationPoint[]
    for name in readdir(input_folder)
        match_result = match(r"^current_Je([0-9.]+)_(up|down|dn)\.h5$", name)
        match_result === nothing && continue

        file_direction = match_result.captures[2] == "dn" ? "down" : match_result.captures[2]
        file_direction == direction || continue

        push!(points, ContinuationPoint(
            parse(Float64, match_result.captures[1]),
            direction,
            joinpath(input_folder, name),
        ))
    end

    isempty(points) && error("No continuation H5 files found for direction '$direction' in $input_folder")

    return if direction == "up"
        sort(points; by = point -> point.Je)
    else
        sort(points; by = point -> point.Je, rev = true)
    end
end

function discover_all_continuation_points(input_folder::String)
    isdir(input_folder) || error("input_folder does not exist or is not a directory: $input_folder")

    points = ContinuationPoint[]
    for direction in ("up", "down")
        try
            append!(points, discover_continuation_points(input_folder, direction))
        catch err
            if !(err isa ErrorException && occursin("No continuation H5 files found for direction", err.msg))
                rethrow()
            end
        end
    end

    isempty(points) && error("No continuation H5 files found in $input_folder")
    return points
end

function continuation_start_index(points::Vector{ContinuationPoint}, start_current::Real)
    isempty(points) && error("Cannot select continuation start from an empty point list")

    if issorted((point.Je for point in points); rev = true)
        index = findfirst(point -> point.Je <= start_current, points)
    else
        index = findlast(point -> point.Je <= start_current, points)
    end

    index === nothing && error("No saved current at or below start=$start_current")
    return index
end

function select_initial_continuation_point(points::Vector{ContinuationPoint}, start_current::Real, preferred_direction::String)
    eligible = filter(point -> point.Je <= start_current, points)
    isempty(eligible) && error("No saved current at or below start=$start_current")

    best_current = maximum(point.Je for point in eligible)
    tied = filter(point -> point.Je == best_current, eligible)
    preferred = findfirst(point -> point.direction == preferred_direction, tied)
    return preferred === nothing ? first(tied) : tied[preferred]
end

function select_target_currents(points::Vector{ContinuationPoint}, direction::String, start_current::Real, end_current::Real)
    initial_point = select_initial_continuation_point(points, start_current, direction)
    unique_currents = sort(unique(point.Je for point in points))

    if direction == "up"
        targets = filter(current -> current >= initial_point.Je && current <= end_current, unique_currents)
    else
        targets = reverse(filter(current -> current <= initial_point.Je && current >= end_current, unique_currents))
    end

    isempty(targets) && error("No continuation targets found between start=$start_current and end=$end_current")
    return initial_point, targets
end

function select_continuation_targets(points::Vector{ContinuationPoint}, start_current::Real, end_current::Real)
    start_index = continuation_start_index(points, start_current)
    start_point = points[start_index]

    if points[1].Je <= points[end].Je
        end_index = findlast(point -> point.Je <= end_current, points)
        end_index === nothing && error("No saved current at or below end=$end_current")
        end_index < start_index && error("end=$end_current is before the selected start current $(start_point.Je)")
    else
        end_index = findlast(point -> point.Je >= end_current, points)
        end_index === nothing && error("No saved current at or above end=$end_current")
        end_index < start_index && error("end=$end_current is before the selected start current $(start_point.Je)")
    end

    return points[start_index:end_index]
end

function continuation_direction(start_current::Real, end_current::Real)
    end_current > start_current && return "up"
    end_current < start_current && return "down"
    error("Continuation requires end != start to determine ramp direction")
end

function resolve_input_folder(config_path::String, input_folder::AbstractString)
    return isabspath(input_folder) ? input_folder : normpath(joinpath(dirname(config_path), input_folder))
end

function parse_continue_args(args::Vector{String})
    config_path = joinpath(@__DIR__, "config.yaml")
    if !isempty(args) && !startswith(args[1], "--")
        config_path = args[1]
        args = args[2:end]
    end

    dry_run = false
    cli_input_folder = nothing
    cli_start = nothing
    cli_end = nothing
    cli_stable = nothing
    cli_ramp = nothing

    idx = 1
    while idx <= length(args)
        arg = args[idx]
        if arg == "--dry-run"
            dry_run = true
            idx += 1
        elseif arg == "--input_folder"
            cli_input_folder = args[idx + 1]
            idx += 2
        elseif arg == "--start"
            cli_start = parse(Float64, args[idx + 1])
            idx += 2
        elseif arg == "--end"
            cli_end = parse(Float64, args[idx + 1])
            idx += 2
        elseif arg == "--stable"
            cli_stable = parse(Float64, args[idx + 1])
            idx += 2
        elseif arg == "--ramptime"
            cli_ramp = parse(Float64, args[idx + 1])
            idx += 2
        else
            error("Unknown argument: $arg")
        end
    end

    isfile(config_path) || error("Config file not found: $config_path")
    cfg = load_config_yaml(config_path)

    raw_input_folder = something(cli_input_folder, get_param(cfg, "sweep", "input_folder", nothing))
    raw_input_folder === nothing && error("Continuation input folder is required")

    raw_start = something(cli_start, get_param(cfg, "continuation_scan", "start", nothing))
    raw_end = something(cli_end, get_param(cfg, "continuation_scan", "end", nothing))
    raw_start === nothing && error("Continuation start current is required")
    raw_end === nothing && error("Continuation end current is required")

    stable_time = Float64(something(cli_stable, get_param(cfg, "continuation_scan", "stable", get_param(cfg, "sweep", "stable_time", PARAM_stable_time))))
    ramp_time = Float64(something(cli_ramp, get_param(cfg, "continuation_scan", "ramptime", PARAM_rampup_time)))

    return ContinuationOptions(
        config_path = abspath(config_path),
        input_folder = resolve_input_folder(config_path, String(raw_input_folder)),
        start_current = Float64(raw_start),
        end_current = Float64(raw_end),
        stable_time = stable_time,
        ramp_time = ramp_time,
        dry_run = dry_run,
    )
end

function build_output_dir()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    output_dir = abspath("sweep_nofastscan_$timestamp")
    mkpath(output_dir)
    return output_dir
end

function save_continuation_config(config_path::String, output_dir::String)
    cp(config_path, joinpath(output_dir, "config.yaml"); force = true)
end

function state_fields_from_vector(u0::Vector, p::TDGLThermalParams)
    cache = TDGLThermalCache(p)
    vector_to_state!(cache.state, u0, p)
    apply_boundary_conditions!(cache.state, p)
    return (
        psi = copy(cache.state.psi),
        Ax = copy(cache.state.Ax),
        Ay = copy(cache.state.Ay),
        T = copy(cache.state.T),
    )
end

function make_state_vector(p::TDGLThermalParams, fields)
    return state_to_vector(initialize_state_from_fields(p, fields.psi, fields.Ax, fields.Ay, fields.T))
end

function ramp_to_current(p_base::TDGLThermalParams, u0::Vector, target_current::Float64, ramp_time::Float64)
    if ramp_time <= 0
        return copy(u0)
    end

    p_current = with_current(p_base, target_current)
    cache = TDGLThermalCache(p_current)
    vector_to_state!(cache.state, u0, p_current)

    prob = ODEProblem(fdm_rhs_thermal!, copy(u0), (0.0, ramp_time), cache)
    integrator = init(prob, Tsit5(), abstol = PARAM_abstol, reltol = PARAM_reltol, save_everystep=false)
    advance_integrator_to!(integrator, ramp_time)
    return copy(integrator.u)
end

@everywhere function run_continuation_stable_worker(p_base::TDGLThermalParams, Je::Float64,
                                                    direction::String, u0_start::Vector,
                                                    stable_time::Float64, dt_snapshots::Float64,
                                                    skip_ratio::Float64, output_dir::String)
    p = with_current(p_base, Je)
    cache = TDGLThermalCache(p)
    vector_to_state!(cache.state, u0_start, p)
    apply_boundary_conditions!(cache.state, p)
    state0 = initialize_state_from_fields(
        p,
        copy(cache.state.psi),
        copy(cache.state.Ax),
        copy(cache.state.Ay),
        copy(cache.state.T),
    )

    h5_file = Printf.format(Printf.Format("current_Je%.4f_%s.h5"), Je, direction)
    h5_path = joinpath(output_dir, h5_file)
    summary = run_stable_simulation_streaming(
        state0, p, stable_time, dt_snapshots, skip_ratio,
        h5_path, Je, direction
    )

    return Dict(
        :J => Je,
        :direction => direction,
        :h5_file => h5_file,
        :V => summary.V_avg,
    )
end

function print_dry_run_summary(direction::String, initial_point::ContinuationPoint, target_currents::Vector{Float64}, output_dir::String)
    target_values = join((@sprintf("%.4f", current) for current in target_currents), ",")
    println("DRY_RUN selected_direction=$direction")
    println("DRY_RUN selected_start=$(@sprintf("%.4f", initial_point.Je))")
    println("DRY_RUN selected_source_direction=$(initial_point.direction)")
    println("DRY_RUN targets=$target_values")
    println("OUTPUT_DIR:$(basename(output_dir))")
end

function run_direction_continuation(p_base::TDGLThermalParams, config::SweepConfig, output_dir::String,
                                    direction::String, initial_point::ContinuationPoint,
                                    target_currents::Vector{Float64},
                                    ramp_time::Float64; dry_run::Bool = false)
    if dry_run
        print_dry_run_summary(direction, initial_point, target_currents, output_dir)
        return Dict[]
    end

    initial_fields = load_continuation_state(initial_point.file)
    initial_params = with_current(p_base, initial_point.Je)
    ramp_state = make_state_vector(initial_params, initial_fields)

    futures = Future[]
    push!(futures, @spawnat :any run_continuation_stable_worker(
        p_base, initial_point.Je, direction, copy(ramp_state),
        config.stable_time, config.dt_snapshots, config.skip_ratio, output_dir
    ))

    for target_current in target_currents[2:end]
        ramp_state = ramp_to_current(p_base, ramp_state, target_current, ramp_time)
        push!(futures, @spawnat :any run_continuation_stable_worker(
            p_base, target_current, direction, copy(ramp_state),
            config.stable_time, config.dt_snapshots, config.skip_ratio, output_dir
        ))
    end

    return sort(fetch.(futures); by = result -> result[:J], rev = direction == "down")
end

function main(args = ARGS)
    println("=" ^ 60)
    println("TDGL + Thermal Continuation Scan")
    println("=" ^ 60)
    ENV["GKSwstype"] = "100"

    options = parse_continue_args(args)
    cfg = load_config_yaml(options.config_path)
    p = build_params_from_config(cfg)
    config = build_sweep_config(cfg)
    config.stable_time = options.stable_time
    config.input_folder = options.input_folder

    output_dir = build_output_dir()
    save_continuation_config(options.config_path, output_dir)

    direction = continuation_direction(options.start_current, options.end_current)
    points = discover_all_continuation_points(options.input_folder)
    initial_point, target_currents = select_target_currents(points, direction, options.start_current, options.end_current)
    run_direction_continuation(p, config, output_dir, direction, initial_point, target_currents, options.ramp_time;
                               dry_run = options.dry_run)

    println("OUTPUT_DIR:$(basename(output_dir))")
    return output_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
