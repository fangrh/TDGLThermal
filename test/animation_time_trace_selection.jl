using Test

ENV["NOFASTSCAN_SKIP_MAIN"] = "1"
include(joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl"))

@testset "active Je time trace selection" begin
    v_t_up = Dict(
        0.1 => [(1.0, 10.0)],
        0.2 => [(2.0, 20.0)],
    )
    v_t_down = Dict(
        0.3 => [(3.0, 30.0)],
    )

    @test select_animation_time_trace(v_t_up, v_t_down, "up", 0.2) == [(2.0, 20.0)]
    @test select_animation_time_trace(v_t_up, v_t_down, "up", 0.5) == Tuple{Float64, Float64}[]
    @test select_animation_time_trace(v_t_up, v_t_down, "down", 0.3) == [(3.0, 30.0)]
end
