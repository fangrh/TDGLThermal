using Test
using OrdinaryDiffEq
import OrdinaryDiffEq: step!

include(joinpath(@__DIR__, "..", "bench", "streaming_iteration_guard.jl"))

mutable struct FakeIntegrator
    t::Float64
    retcode
    dt::Float64
    opts
end

struct FakeOpts
    maxiters::Int
end

function step!(x::FakeIntegrator)
    x.retcode == ReturnCode.MaxIters && return x.retcode
    x.t += x.dt
    x.retcode = ReturnCode.Success
    return x.retcode
end

@testset "streaming iteration guard" begin
    @test checked_stream_step!(FakeIntegrator(0.0, ReturnCode.Default, 1.0, FakeOpts(3))) == ReturnCode.Success
    @test_throws ErrorException checked_stream_step!(FakeIntegrator(0.0, ReturnCode.MaxIters, 1.0, FakeOpts(3)))
    @test_throws ErrorException checked_stream_step!(FakeIntegrator(0.0, ReturnCode.Default, 0.0, FakeOpts(3)))
end

@testset "maxiters override parsing" begin
    old = get(ENV, "TDGL_STREAM_MAXITERS", nothing)
    try
        delete!(ENV, "TDGL_STREAM_MAXITERS")
        @test stream_maxiters_override() === nothing
        ENV["TDGL_STREAM_MAXITERS"] = "100"
        @test stream_maxiters_override() == 100
        for bad in ("0", "-1", "1.5", "abc")
            ENV["TDGL_STREAM_MAXITERS"] = bad
            @test_throws ErrorException stream_maxiters_override()
        end
    finally
        old === nothing ? delete!(ENV, "TDGL_STREAM_MAXITERS") : (ENV["TDGL_STREAM_MAXITERS"] = old)
    end
end

@testset "real OrdinaryDiffEq maxiters guard" begin
    rhs!(du, u, p, t) = (du[1] = 1.0)
    prob = ODEProblem(rhs!, [0.0], (0.0, 10.0))
    limited = init(prob, Tsit5(); abstol=1e-8, reltol=1e-8,
                   save_everystep=false, maxiters=3)
    withenv("TDGL_STREAM_MAXITERS" => nothing) do
        @test_throws ErrorException advance_integrator_to!(limited, 10.0)
        @test limited.sol.retcode == ReturnCode.MaxIters
    end
    withenv("TDGL_STREAM_MAXITERS" => "100") do
        overridden = init(prob, Tsit5(); abstol=1e-8, reltol=1e-8,
                          save_everystep=false, maxiters=3)
        for target in (0.0, 0.0, .1, .2, 1.0, 1.0, 5.0, 10.0)
            advance_integrator_to!(overridden, target)
            @test overridden.t == target
            @test overridden.u[1] ≈ target atol=1e-12
        end
        @test overridden.t >= 10.0
        @test overridden.opts.maxiters == 100
    end
end
