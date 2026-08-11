# Full-Curl TDGL runner with rolling single-frame checkpoints.
#
# This wrapper deliberately leaves current_sweep_thermal_nofastscan.jl
# byte-identical to the reviewer-verified Full Curl solver. It replaces only
# the streamed persistence function and then invokes the original main().

include(joinpath(@__DIR__, "current_sweep_thermal_nofastscan.jl"))

@everywhere begin

function rolling_checkpoint_interval()
    interval = parse(Float64, get(ENV, "TDGL_CHECKPOINT_INTERVAL", "0"))
    interval >= 0.0 || error("TDGL_CHECKPOINT_INTERVAL must be non-negative")
    return interval
end

function write_rolling_checkpoint(checkpoint_path::String, state::TDGLThermalState,
                                  checkpoint_time::Float64, stable_time::Float64,
                                  Je::Float64, direction::String)
    temporary_path = checkpoint_path * ".tmp"

    h5open(temporary_path, "w") do file
        file["Je"] = Je
        file["direction"] = direction
        file["checkpoint_time"] = checkpoint_time
        file["target_stable_time"] = stable_time
        file["times"] = [checkpoint_time]
        file["psi_real"] = reshape(real.(state.psi), size(state.psi)..., 1)
        file["psi_imag"] = reshape(imag.(state.psi), size(state.psi)..., 1)
        file["T"] = reshape(state.T, size(state.T)..., 1)
        file["Ax"] = reshape(state.Ax, size(state.Ax)..., 1)
        file["Ay"] = reshape(state.Ay, size(state.Ay)..., 1)
    end

    # Atomic replacement keeps the preceding complete checkpoint if a job is
    # interrupted while the next temporary file is being written.
    mv(temporary_path, checkpoint_path; force=true)
    return checkpoint_path
end

function run_stable_simulation_streaming(state0::TDGLThermalState, p::TDGLThermalParams,
                                         stable_time::Float64, dt_snapshots::Float64,
                                         skip_ratio::Float64, h5_path::String,
                                         Je::Float64, direction::String)
    checkpoint_interval = rolling_checkpoint_interval()
    cache = TDGLThermalCache(p)
    u0 = state_to_vector(state0)
    tspan = (0.0, stable_time)
    prob = ODEProblem(fdm_rhs_thermal!, u0, tspan, cache)
    integrator = init(prob, Tsit5(), abstol=1e-4, reltol=1e-4, save_everystep=false)

    snapshot_times = collect(0.0:dt_snapshots:stable_time)
    if isempty(snapshot_times) || snapshot_times[end] != stable_time
        push!(snapshot_times, stable_time)
    end

    original_n_snap = length(snapshot_times)
    skip_idx = max(1, ceil(Int, original_n_snap * skip_ratio))
    kept_count = 0
    V_sum = 0.0

    buffers = initialize_stream_buffers(p, STREAM_WRITE_BATCH_SIZE)
    buffer_count = 0
    next_store_idx = 1
    du = similar(u0)
    checkpoint_path = joinpath(dirname(h5_path),
                               Printf.format(Printf.Format("checkpoint_Je%.4f_%s.h5"), Je, direction))
    next_checkpoint_time = checkpoint_interval > 0.0 ? checkpoint_interval : Inf

    h5open(h5_path, "w") do file
        file["Je"] = Je
        file["direction"] = direction
        file["stored_skip_idx"] = skip_idx
        file["original_n_snap"] = original_n_snap
        datasets = create_extendable_snapshot_datasets(file, p, STREAM_WRITE_BATCH_SIZE)

        for (snapshot_idx, target_t) in enumerate(snapshot_times)
            if snapshot_idx > 1
                advance_integrator_to!(integrator, target_t)
            end

            checkpoint_due = target_t + eps(target_t) >= next_checkpoint_time && target_t < stable_time
            snapshot_due = snapshot_idx >= skip_idx
            if !checkpoint_due && !snapshot_due
                continue
            end

            vector_to_state!(cache.state, integrator.u, p)
            apply_boundary_conditions!(cache.state, p)

            if checkpoint_due
                write_rolling_checkpoint(checkpoint_path, cache.state, integrator.t,
                                         stable_time, Je, direction)
                println("CHECKPOINT Je=$(Je) direction=$(direction) time=$(integrator.t) path=$(checkpoint_path)")
                flush(stdout)
                while next_checkpoint_time <= target_t
                    next_checkpoint_time += checkpoint_interval
                end
            end

            if !snapshot_due
                continue
            end

            buffer_count += 1
            fdm_rhs_thermal!(du, integrator.u, cache, integrator.t)

            buffers.times[buffer_count] = integrator.t
            buffers.V[buffer_count] = compute_voltage_from_du(du, p)
            buffers.psi_avg[buffer_count] = mean(abs2.(cache.state.psi[2:end-1, 2:end-1]))
            buffers.T_avg[buffer_count] = mean(cache.state.T[2:end-1, 2:end-1])
            buffers.psi_real[:, :, buffer_count] .= real.(cache.state.psi)
            buffers.psi_imag[:, :, buffer_count] .= imag.(cache.state.psi)
            buffers.T[:, :, buffer_count] .= cache.state.T
            buffers.Ax[:, :, buffer_count] .= cache.state.Ax
            buffers.Ay[:, :, buffer_count] .= cache.state.Ay

            kept_count += 1
            V_sum += buffers.V[buffer_count]

            if buffer_count == STREAM_WRITE_BATCH_SIZE
                next_store_idx = append_snapshot_batch!(datasets, buffers, next_store_idx, buffer_count)
                buffer_count = 0
            end
        end

        if buffer_count > 0
            append_snapshot_batch!(datasets, buffers, next_store_idx, buffer_count)
        end
    end

    return (
        V_avg = kept_count > 0 ? V_sum / kept_count : 0.0,
        stored_skip_idx = skip_idx,
        original_n_snap = original_n_snap,
    )
end

end # @everywhere

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    output_dir = main()
    println("OUTPUT_DIR:$output_dir")
end
