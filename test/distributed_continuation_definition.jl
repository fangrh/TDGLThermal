using Test
using HDF5

include(joinpath(@__DIR__, "..", "current_sweep_thermal_continue.jl"))

@testset "continuation scan target selection" begin
    mktempdir() do dir
        touch(joinpath(dir, "current_Je0.1000_up.h5"))
        touch(joinpath(dir, "current_Je0.2000_up.h5"))
        touch(joinpath(dir, "current_Je0.3000_up.h5"))
        touch(joinpath(dir, "current_Je0.4000_up.h5"))
        touch(joinpath(dir, "current_Je0.4000_down.h5"))
        touch(joinpath(dir, "current_Je0.3000_down.h5"))
        touch(joinpath(dir, "current_Je0.2000_down.h5"))
        touch(joinpath(dir, "current_Je0.5000_down.h5"))

        points = discover_continuation_points(dir, "up")
        targets = select_continuation_targets(points, 0.21, 0.44)
        down_points = discover_continuation_points(dir, "down")
        down_targets = select_continuation_targets(down_points, 0.39, 0.09)
        all_points = discover_all_continuation_points(dir)
        initial_up_from_down, mixed_up_targets = select_target_currents(all_points, "up", 0.39, 0.44)

        @test [point.Je for point in points] == [0.1, 0.2, 0.3, 0.4]
        @test continuation_start_index(points, 0.21) == 2
        @test [point.Je for point in targets] == [0.2, 0.3, 0.4]
        @test continuation_direction(0.1, 0.2) == "up"
        @test continuation_direction(0.2, 0.1) == "down"
        @test_throws ErrorException continuation_direction(0.2, 0.2)
        @test [point.Je for point in down_points] == [0.5, 0.4, 0.3, 0.2]
        @test continuation_start_index(down_points, 0.39) == 3
        @test [point.Je for point in down_targets] == [0.3, 0.2]
        @test initial_up_from_down.Je == 0.3
        @test initial_up_from_down.direction == "up"
        @test mixed_up_targets == [0.3, 0.4]
    end
end

@testset "voltage trace uses dA/dt" begin
    p = TDGLThermalParams(Nx = 4, Ny = 4, hx = 0.5, hy = 0.5, Je = 0.1)
    state = initialize_state_with_params(p, 1.0, 0.0)
    u0 = state_to_vector(state)
    u_vectors = [copy(u0), copy(u0)]

    trace = compute_voltage_trace_from_solution_vectors(u_vectors, p)
    du = similar(u0)
    cache = TDGLThermalCache(p)
    fdm_rhs_thermal!(du, u0, cache, 0.0)

    @test length(trace) == 2
    @test trace[1] == compute_voltage_from_du(du, p)
    @test trace[2] == compute_voltage_from_du(du, p)
end

@testset "stable simulation returns voltage trace" begin
    p = TDGLThermalParams(Nx = 4, Ny = 4, hx = 0.5, hy = 0.5, Je = 0.1)
    state = initialize_state_with_params(p, 1.0, 0.0)

    psi_res, Ax_res, Ay_res, T_res, times, V_trace = run_stable_simulation(state, p, 0.05, 0.05)

    @test size(psi_res, 3) == length(times) == length(V_trace)
    @test size(Ax_res, 3) == length(times)
    @test size(Ay_res, 3) == length(times)
    @test size(T_res, 3) == length(times)
end

@testset "stable output file keeps trimmed layout" begin
    mktempdir() do dir
        p = TDGLThermalParams(Nx = 4, Ny = 4, hx = 0.5, hy = 0.5)
        config = SweepConfig(
            0.2, 2, 1.0, 1.0,
            0.10, 0.05, 0.5, true, false,
            false, 5, 20,
            1.0, 0.0, 0.0, 1.2,
            nothing,
        )

        result = simulate_point(0.1, p, config, "up", dir)
        h5_path = joinpath(dir, result[:h5_file])

        h5open(h5_path, "r") do file
            times = read(file, "times")
            voltage = read(file, "V")
            psi_real = read(file, "psi_real")
            temperature = read(file, "T")

            @test haskey(file, "times")
            @test haskey(file, "V")
            @test haskey(file, "psi_real")
            @test haskey(file, "psi_imag")
            @test haskey(file, "T")
            @test haskey(file, "Ax")
            @test haskey(file, "Ay")
            @test haskey(file, "stored_skip_idx")
            @test haskey(file, "original_n_snap")
            @test length(times) == length(voltage)
            @test size(psi_real, 3) == length(times)
            @test size(temperature, 3) == length(times)
            @test read(file, "stored_skip_idx") == 2
            @test read(file, "original_n_snap") == 3
            @test times == [0.05, 0.10]
            @test all(diff(times) .> 0)
        end
    end
end
