using Test
using HDF5

ENV["NOFASTSCAN_SKIP_WORKERS"] = "1"
ENV["TDGL_CHECKPOINT_INTERVAL"] = "0.05"
include(joinpath(@__DIR__, "..", "current_sweep_thermal_checkpoint.jl"))

@testset "rolling checkpoint keeps one atomic state" begin
    mktempdir() do dir
        p = TDGLThermalParams(Nx=4, Ny=4, hx=0.5, hy=0.5, Je=0.1)
        state = initialize_state_with_params(p, 1.0, 0.0)
        output_path = joinpath(dir, "current_Je0.1000_up.h5")

        run_stable_simulation_streaming(
            state, p, 0.10, 0.05, 0.5, output_path, 0.1, "up"
        )

        checkpoint_path = joinpath(dir, "checkpoint_Je0.1000_up.h5")
        @test isfile(checkpoint_path)
        @test !isfile(checkpoint_path * ".tmp")

        h5open(checkpoint_path, "r") do file
            @test read(file, "checkpoint_time") == 0.05
            @test read(file, "target_stable_time") == 0.10
            @test read(file, "times") == [0.05]
            @test size(read(file, "psi_real"), 3) == 1
            @test size(read(file, "psi_imag"), 3) == 1
            @test size(read(file, "T"), 3) == 1
            @test size(read(file, "Ax"), 3) == 1
            @test size(read(file, "Ay"), 3) == 1
        end

        h5open(output_path, "r") do file
            @test read(file, "times") == [0.05, 0.10]
        end
    end
end
