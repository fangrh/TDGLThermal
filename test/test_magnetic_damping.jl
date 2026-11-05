using Test

ENV["NOFASTSCAN_SKIP_MAIN"] = "1"
ENV["NOFASTSCAN_SKIP_WORKERS"] = "1"
include(joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl"))

@testset "interior transverse magnetic damping and constant mode" begin
    p = TDGLThermalParams(Nx=17, Ny=17, hx=0.5, hy=0.5,
                          kappa=2.0, sigma=1.0, Je=0.0, He=0.0)
    c = TDGLThermalCache(p)
    fill!(c.state.Ay, 0.0)
    for j in axes(c.state.Ax, 2), i in axes(c.state.Ax, 1)
        c.state.Ax[i,j] = sin(2pi * (j-1) / p.Ny)
    end
    fill!(c.Jsx, 0.0)
    fill!(c.Jsy, 0.0)
    compute_dA_dt!(c.dAx, c.dAy, c.state, p, c.Jsx, c.Jsy, c.sigma)
    work = sum(c.state.Ax[2:end-1, 2:end-1] .* c.dAx[2:end-1, 2:end-1])
    @test work < 0
    fill!(c.state.Ax, 1.0)
    compute_dA_dt!(c.dAx, c.dAy, c.state, p, c.Jsx, c.Jsy, c.sigma)
    @test maximum(abs, c.dAx) == 0
    @test maximum(abs, c.dAy) == 0
end
