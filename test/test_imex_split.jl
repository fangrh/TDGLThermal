using Test
using SparseArrays
using Random

ENV["NOFASTSCAN_SKIP_MAIN"] = "1"
ENV["NOFASTSCAN_SKIP_WORKERS"] = "1"
include(joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl"))
if !isdefined(Main, :fdm_rhs_compatible!)
    include(joinpath(@__DIR__, "..", "src", "cpu_compatible.jl"))
end
include(joinpath(@__DIR__, "..", "src", "imex_experiment.jl"))

@testset "deterministic IMEX spatial split" begin
    @test isdefined(Main, :build_spatial_split)
    function fixture_params(; bc_x=:periodic, bc_y=:neumann, coverage=.88,
                            loop=false, field_side=:both, thermal_enabled=true)
        TDGLThermalParams(Nx=4, Ny=3, hx=.5, hy=.75, kappa=2.,
            sigma1=.7, sigma2=1., sigma3=1.4, Tc1gap=.2, Tc3gap=.2,
            Je=.12, He=.03, bc_x=bc_x, bc_y=bc_y, bc_y_coverage=coverage,
            loop=loop, field_side=field_side, thermal_enabled=thermal_enabled,
            noise_strength=0., T0=.2, k_eff=5.)
    end
    function fixture_state(p)
        Random.seed!(20261104)
        s = initialize_state_clean(p)
        s.psi .= complex.(.3 .* randn(size(s.psi)), .3 .* randn(size(s.psi)))
        s.Ax .= .1 .* randn(size(s.Ax)); s.Ay .= .1 .* randn(size(s.Ay))
        s.T .= .2 .+ .2 .* rand(size(s.T)...)
        s
    end
    for cfg in ((;bc_x=:periodic,bc_y=:neumann,coverage=.88,loop=false,field_side=:both),
                (;bc_x=:dirichlet,bc_y=:neumann,coverage=1.,loop=false,field_side=:both),
                (;bc_x=:neumann,bc_y=:neumann,coverage=.5,loop=false,field_side=:upper),
                (;bc_x=:periodic,bc_y=:dirichlet,coverage=.88,loop=false,field_side=:lower),
                (;bc_x=:dirichlet,bc_y=:dirichlet,coverage=1.,loop=false,field_side=:both),
                (;bc_x=:neumann,bc_y=:dirichlet,coverage=.5,loop=true,field_side=:both),
                (;bc_x=:periodic,bc_y=:neumann,coverage=.88,loop=false,field_side=:both,
                 thermal_enabled=false))
        p = fixture_params(; cfg...)
        split = build_spatial_split(p)
        s = fixture_state(p); u = state_to_vector(s)
        g = similar(u); isolated_spatial!(g, u, split.cache)
        @test maximum(abs, g - (split.L * u + split.affine)) /
              (1 + maximum(abs, g)) ≤ 1e-11
        # A second probe amplitude must give the same affine map.
        uhalf = .5 .* u; ghalf = similar(u)
        isolated_spatial!(ghalf, uhalf, split.cache)
        @test maximum(abs, ghalf - (split.L * uhalf + split.affine)) /
              (1 + maximum(abs, ghalf)) ≤ 1e-11
        split_half = build_spatial_split(p; h=.5)
        @test maximum(abs, split_half.affine - split.affine) ≤ 1e-12
        @test maximum(abs, (split_half.L - split.L).nzval; init=0) ≤ 1e-12
        # ψ backgrounds are irrelevant when Js and W are explicitly zero.
        s2 = fixture_state(p); s2.psi .*= (3 + 2im)
        u2 = state_to_vector(s2); g2 = similar(u2)
        isolated_spatial!(g2, u2, split.cache)
        @test g2 == g
    end
end

@testset "IMEX residual recombination and physical subspace" begin
    p = TDGLThermalParams(Nx=4, Ny=3, hx=.5, hy=.75, kappa=2.,
        sigma1=.7, sigma2=1., sigma3=1.4, Je=.12, He=.03,
        noise_strength=0., T0=.2, k_eff=5.)
    split = build_spatial_split(p)
    ctx = ImexContext(split)
    s = initialize_state_clean(p)
    Random.seed!(20261104)
    s.psi .= complex.(.3 .* randn(size(s.psi)), .3 .* randn(size(s.psi)))
    s.Ax .= .1 .* randn(size(s.Ax)); s.Ay .= .1 .* randn(size(s.Ay))
    s.T .= .2 .+ .2 .* rand(size(s.T)...)
    u = state_to_vector(s); original = copy(u)
    full = similar(u); residual = similar(u)
    fdm_rhs_compatible!(full, u, TDGLThermalCache(p), 0.)
    imex_residual!(residual, u, ctx, 0.)
    reconstructed = split.L * u + residual
    @test u == original
    @test maximum(abs, reconstructed - full) / (1 + maximum(abs, full)) ≤ 1e-11
    nψ = (p.Nx + 1) * (p.Ny + 1)
    @test all(iszero, imag.(split.affine[nψ+1:end]))
    @test all(iszero, imag.(split.L[nψ+1:end, :]))
    @test all(isfinite, real.(split.affine)) && all(isfinite, real.(split.L.nzval))

    # Affine boundary forcing changes the intercept, not the assembled slope.
    p0 = TDGLThermalParams(p.Nx, p.Ny, p.hx, p.hy, p.dt, p.kappa, p.sigma1,
        p.sigma2, p.sigma3, p.u, p.gamma, p.C_eff, p.k_eff, p.T0, p.eta_env,
        p.eta_hole, p.holewidth, 0., 0., p.bc_x, p.bc_y, p.bc_y_coverage,
        p.loop, p.field_side, p.Tc1, p.Tc2, p.Tc3, p.Tc1gap, p.Tc3gap,
        p.psi_y_zero, p.thermal_enabled, p.epsilon_type, p.noise_strength)
    s0 = build_spatial_split(p0)
    @test maximum(abs, s0.L - split.L) ≤ 1e-12
    @test maximum(abs, s0.affine - split.affine) > 0
end
