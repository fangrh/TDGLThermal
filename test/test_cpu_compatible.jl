using Test
using Random

# Keep reference imports side-effect free in this focused test.
ENV["NOFASTSCAN_SKIP_MAIN"] = "1"
ENV["NOFASTSCAN_SKIP_WORKERS"] = "1"
include(joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl"))

# The controller may include this file as part of the solver.  In a standalone
# test invocation load it exactly once; on the red-test commit this leaves the
# required compatible entry points undefined.
if !isdefined(Main, :fdm_rhs_compatible!)
    candidate = joinpath(@__DIR__, "..", "src", "cpu_compatible.jl")
    isfile(candidate) && include(candidate)
end

const FIXTURE_SEEDS = (20261104, 20261105, 20261106, 20261107)
const FIXTURES = (
    (; bc_x=:periodic,  bc_y=:neumann,   coverage=0.88, loop=false, field_side=:both),
    (; bc_x=:dirichlet, bc_y=:neumann,   coverage=1.00, loop=false, field_side=:both),
    (; bc_x=:neumann,   bc_y=:neumann,   coverage=0.50, loop=false, field_side=:upper),
    (; bc_x=:periodic,  bc_y=:dirichlet, coverage=0.88, loop=false, field_side=:lower),
    (; bc_x=:dirichlet, bc_y=:dirichlet, coverage=1.00, loop=false, field_side=:both),
    (; bc_x=:neumann,   bc_y=:dirichlet, coverage=0.50, loop=false, field_side=:both),
    (; bc_x=:periodic,  bc_y=:neumann,   coverage=0.50, loop=true,  field_side=:both),
    (; bc_x=:periodic,  bc_y=:neumann,   coverage=0.88, loop=false, field_side=:both,
      psi_y_zero=true),
    (; bc_x=:periodic,  bc_y=:neumann,   coverage=0.88, loop=false, field_side=:both,
      epsilon_type=:bcs),
    (; bc_x=:periodic,  bc_y=:neumann,   coverage=0.88, loop=false, field_side=:both,
      thermal_enabled=false),
)

function fixture_params(f; noise_strength=0.0)
    TDGLThermalParams(Nx=7, Ny=5, hx=0.5, hy=0.75, kappa=2.0,
        sigma1=0.7, sigma2=1.0, sigma3=1.4, Tc1gap=0.2, Tc3gap=0.2,
        Je=0.12, He=0.03, bc_x=f.bc_x, bc_y=f.bc_y,
        bc_y_coverage=f.coverage, loop=f.loop, field_side=f.field_side,
        psi_y_zero=get(f, :psi_y_zero, false),
        epsilon_type=get(f, :epsilon_type, :linear),
        thermal_enabled=get(f, :thermal_enabled, true),
        noise_strength=noise_strength, T0=0.2, k_eff=5.0)
end

function fixture_state(p, seed)
    Random.seed!(seed)
    s = initialize_state_clean(p)
    s.psi .= complex.(0.3 .* randn(size(s.psi)), 0.3 .* randn(size(s.psi)))
    s.Ax .= 0.1 .* randn(size(s.Ax))
    s.Ay .= 0.1 .* randn(size(s.Ay))
    s.T .= 0.2 .+ 0.2 .* rand(size(s.T)...)
    return s
end

function rhs_pair(p, seed; noise_strength=0.0)
    p = TDGLThermalParams(p.Nx, p.Ny, p.hx, p.hy, p.dt, p.kappa, p.sigma1,
        p.sigma2, p.sigma3, p.u, p.gamma, p.C_eff, p.k_eff, p.T0, p.eta_env,
        p.eta_hole, p.holewidth, p.Je, p.He, p.bc_x, p.bc_y, p.bc_y_coverage,
        p.loop, p.field_side, p.Tc1, p.Tc2, p.Tc3, p.Tc1gap, p.Tc3gap,
        p.psi_y_zero, p.thermal_enabled, p.epsilon_type, noise_strength)
    u = state_to_vector(fixture_state(p, seed))
    du_ref = similar(u)
    du_fast = similar(u)
    cref = TDGLThermalCache(p)
    cfast = TDGLThermalCache(p)
    Random.seed!(seed)
    fdm_rhs_thermal!(du_ref, u, cref, 0.0)
    tail_ref = randn(16)
    Random.seed!(seed)
    fdm_rhs_compatible!(du_fast, u, cfast, 0.0)
    tail_fast = randn(16)
    return du_ref, du_fast, tail_ref, tail_fast
end

function _normalized_field_error(du_fast, du_ref, r)
    maxdiff = 0.0
    maxref = 0.0
    @inbounds for k in r
        maxdiff = max(maxdiff, abs(du_fast[k] - du_ref[k]))
        maxref = max(maxref, abs(du_ref[k]))
    end
    return maxdiff / (1 + maxref)
end

function field_errors(du_fast, du_ref, p)
    npsi = (p.Nx + 1) * (p.Ny + 1)
    nax = p.Nx * (p.Ny + 1)
    nay = (p.Nx + 1) * p.Ny
    ranges = (1:npsi, npsi + 1:npsi + nax,
              npsi + nax + 1:npsi + nax + nay,
              npsi + nax + nay + 1:length(du_ref))
    return map(r -> _normalized_field_error(du_fast, du_ref, r), ranges)
end

@testset "compatible CPU RHS finite fixture matrix" begin
    for (idx, f) in enumerate(FIXTURES), seed in FIXTURE_SEEDS
        for noise in (0.0, 0.001)
            p = fixture_params(f; noise_strength=noise)
            du_ref, du_fast, tail_ref, tail_fast = rhs_pair(p, seed; noise_strength=noise)
            @test all(≤(1e-11), field_errors(du_fast, du_ref, p))
            @test tail_fast == tail_ref
        end
    end
end

@testset "compatible link-variable update regression" begin
    for (nx, ny) in ((7, 5), (60, 30))
        p = TDGLThermalParams(Nx=nx, Ny=ny, hx=0.5, hy=0.75)
        Random.seed!(20261104 + nx)
        sref = initialize_state_clean(p)
        sfast = initialize_state_clean(p)
        sref.Ax .= sfast.Ax .= 1e-14 .* randn(size(sref.Ax))
        sref.Ay .= sfast.Ay .= 1e-14 .* randn(size(sref.Ay))
        # Include ordinary and large phases in addition to near-zero values.
        sref.Ax[1] = sfast.Ax[1] = 0.37
        sref.Ax[end] = sfast.Ax[end] = 1.0e4
        sref.Ay[1] = sfast.Ay[1] = -0.41
        sref.Ay[end] = sfast.Ay[end] = -1.0e4
        update_link_variables!(sref, p)
        update_link_variables_compatible!(sfast, p)
        maxdiff = max(maximum(abs, sfast.Ux .- sref.Ux),
                      maximum(abs, sfast.Uy .- sref.Uy))
        exact = sfast.Ux == sref.Ux && sfast.Uy == sref.Uy
        @info "link-variable regression" nx ny maxdiff exact
        @test maxdiff <= 1e-15
    end
end

@testset "compatible backend selection" begin
    @test resolve_rhs_backend("reference") === fdm_rhs_thermal!
    @test resolve_rhs_backend("cpu_compatible") === fdm_rhs_compatible!
    old = get(ENV, "TDGL_RHS_BACKEND", nothing)
    try
        ENV["TDGL_RHS_BACKEND"] = "cpu_compatible"
        @test resolve_rhs_backend() === fdm_rhs_compatible!
        ENV["TDGL_RHS_BACKEND"] = "reference"
        @test resolve_rhs_backend() === fdm_rhs_thermal!
        @test_throws ArgumentError resolve_rhs_backend("does-not-exist")
    finally
        old === nothing ? delete!(ENV, "TDGL_RHS_BACKEND") : (ENV["TDGL_RHS_BACKEND"] = old)
    end
end

@testset "compatible backend allocation trend" begin
    for (nx, ny) in ((60, 30), (120, 60))
        p = TDGLThermalParams(Nx=nx, Ny=ny, hx=0.5, hy=0.5, kappa=2.0,
            sigma1=0.7, sigma2=1.0, sigma3=1.4, Tc1gap=0.2, Tc3gap=0.2,
            Je=0.12, He=0.03, noise_strength=0.0, T0=0.2, k_eff=5.0)
        u = state_to_vector(fixture_state(p, 20261104))
        cref, cfast = TDGLThermalCache(p), TDGLThermalCache(p)
        dr, df = similar(u), similar(u)
        fdm_rhs_thermal!(dr, u, cref, 0.0)
        fdm_rhs_compatible!(df, u, cfast, 0.0)
        aref = @allocated fdm_rhs_thermal!(dr, u, cref, 0.0)
        afast = @allocated fdm_rhs_compatible!(df, u, cfast, 0.0)
        @test afast < aref
    end
end
