using Test

ENV["NOFASTSCAN_SKIP_MAIN"] = "1"
include(joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl"))

@testset "vector-potential equation implements minus curl curl A" begin
    p = TDGLThermalParams(Nx = 7, Ny = 7, hx = 0.5, hy = 0.75, kappa = 2.0, Je = 0.0, He = 0.0)
    cache = TDGLThermalCache(p)

    ax_field(x, y) = x^3 + 2x^2 * y + 3x * y^2 + 4y^3
    ay_field(x, y) = 5x^3 + 6x^2 * y + 7x * y^2 + 8y^3

    state = cache.state
    for j in 1:p.Ny+1
        for i in 1:p.Nx
            x = (i - 1) * p.hx
            y = (j - 1) * p.hy
            state.Ax[i, j] = ax_field(x, y)
        end
    end
    for j in 1:p.Ny
        for i in 1:p.Nx+1
            x = (i - 1) * p.hx
            y = (j - 1) * p.hy
            state.Ay[i, j] = ay_field(x, y)
        end
    end

    fill!(cache.Jsx, 0.0)
    fill!(cache.Jsy, 0.0)
    compute_dA_dt!(cache.dAx, cache.dAy, state, p, cache.Jsx, cache.Jsy, cache.sigma)

    expected_dax(i, j) = -p.kappa^2 * (
        (
            (ay_field(i * p.hx, (j - 1) * p.hy) - ay_field((i - 1) * p.hx, (j - 1) * p.hy)) -
            (ay_field(i * p.hx, (j - 2) * p.hy) - ay_field((i - 1) * p.hx, (j - 2) * p.hy))
        ) / (p.hx * p.hy) -
        (
            ax_field((i - 1) * p.hx, j * p.hy) - 2 * ax_field((i - 1) * p.hx, (j - 1) * p.hy) +
            ax_field((i - 1) * p.hx, (j - 2) * p.hy)
        ) / (p.hy^2)
    )
    expected_day(i, j) = -p.kappa^2 * (
        (
            (ax_field((i - 1) * p.hx, j * p.hy) - ax_field((i - 2) * p.hx, j * p.hy)) -
            (ax_field((i - 1) * p.hx, (j - 1) * p.hy) - ax_field((i - 2) * p.hx, (j - 1) * p.hy))
        ) / (p.hx * p.hy) -
        (
            ay_field(i * p.hx, (j - 1) * p.hy) - 2 * ay_field((i - 1) * p.hx, (j - 1) * p.hy) +
            ay_field((i - 2) * p.hx, (j - 1) * p.hy)
        ) / (p.hx^2)
    )

    for (i, j) in ((3, 3), (5, 4))
        @test isapprox(cache.dAx[i, j], expected_dax(i, j); atol = 1e-10)
        @test isapprox(cache.dAy[i, j], expected_day(i, j); atol = 1e-10)
    end
end
