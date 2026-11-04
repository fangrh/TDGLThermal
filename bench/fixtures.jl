# Deterministic fixtures and case construction for the CPU harness.

using SHA
using Random
using Serialization

if !isdefined(Main, :TDGLThermalParams)
    ENV["NOFASTSCAN_SKIP_MAIN"] = "1"
    ENV["NOFASTSCAN_SKIP_WORKERS"] = "1"
    include(joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl"))
end

const BENCH_MODEL_FILE = normpath(joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl"))

function _bytes_hash(x)
    io = IOBuffer(); serialize(io, x)
    bytes2hex(SHA.sha256(take!(io)))
end

function _state_hash(state)
    _bytes_hash((state.psi, state.Ax, state.Ay, state.T))
end

"""Construct a reproducible case. `preparation` is :cool, :hot, or :random."""
function make_case(config_path::String; Je::Float64, noise::Float64, seed::Int,
                   preparation::Symbol, Nx::Int=60, Ny::Int=30,
                   initial_state_path::Union{Nothing,String}=nothing)
    preparation in (:cool, :hot, :random, :configured_down) ||
        throw(ArgumentError("unknown preparation: $preparation"))
    cfg = load_config_yaml(config_path)
    p0 = build_params_from_config(cfg)
    # Keep the physical domain fixed when changing the grid.
    xspan, yspan = p0.Nx * p0.hx, p0.Ny * p0.hy
    p = TDGLThermalParams(Nx=Nx, Ny=Ny, hx=xspan/Nx, hy=yspan/Ny,
        kappa=p0.kappa, sigma1=p0.sigma1, sigma2=p0.sigma2, sigma3=p0.sigma3,
        u=p0.u, gamma=p0.gamma, C_eff=p0.C_eff, k_eff=p0.k_eff, T0=p0.T0,
        eta_env=p0.eta_env, eta_hole=p0.eta_hole, holewidth=p0.holewidth,
        Je=Je, He=p0.He, bc_x=p0.bc_x, bc_y=p0.bc_y,
        bc_y_coverage=p0.bc_y_coverage, loop=p0.loop, field_side=p0.field_side,
        Tc1=p0.Tc1, Tc2=p0.Tc2, Tc3=p0.Tc3, Tc1gap=p0.Tc1gap,
        Tc3gap=p0.Tc3gap, psi_y_zero=p0.psi_y_zero,
        thermal_enabled=p0.thermal_enabled, epsilon_type=p0.epsilon_type,
        noise_strength=noise)
    Random.seed!(seed)
    shape = (Nx+1, Ny+1)
    psi_scale, temp_base = preparation === :hot ? (0.05, max(p.T0, 1.2)) :
        preparation === :cool ? (0.8, p.T0) : (0.3, p.T0 + 0.1)
    psi = ComplexF64.(psi_scale .* randn(shape), psi_scale .* randn(shape))
    Ax = 0.1 .* randn(Nx, Ny+1)
    Ay = 0.1 .* randn(Nx+1, Ny)
    T = temp_base .+ 0.05 .* rand(shape...)
    state = if preparation === :configured_down
        sweep = build_sweep_config(cfg)
        initialize_state_with_params(p, sweep.psi_down_init, sweep.T_initial_down)
    else
        initialize_state_from_fields(p, psi, Ax, Ay, T)
    end
    if initial_state_path !== nothing
        state = h5open(initial_state_path, "r") do file
            isapprox(read(file["Je"]), Je; atol=1e-12, rtol=0) ||
                error("continuation current differs from requested current")
            psi_real = read(file["psi_real"])
            psi_imag = read(file["psi_imag"])
            ax = read(file["Ax"]); ay = read(file["Ay"]); temp = read(file["T"])
            initialize_state_from_fields(p, complex.(psi_real[:,:,end], psi_imag[:,:,end]),
                ax[:,:,end], ay[:,:,end], temp[:,:,end])
        end
    end
    all(isfinite, state_to_vector(state)) || error("nonfinite initial state")
    return (params=p, state=state, u0=state_to_vector(state), config_path=abspath(config_path),
            Je=Je, noise=noise, seed=seed, preparation=preparation,
            initial_hash=_state_hash(state), model_hash=bytes2hex(SHA.sha256(read(BENCH_MODEL_FILE))),
            rhs_source_hash=isfile(joinpath(@__DIR__, "..", "src", "cpu_compatible.jl")) ?
                bytes2hex(SHA.sha256(read(joinpath(@__DIR__, "..", "src", "cpu_compatible.jl")))) : "missing",
            config_hash=isfile(config_path) ? bytes2hex(SHA.sha256(read(config_path))) : "missing",
            julia_version=string(VERSION), machine=Sys.MACHINE,
            initial_state_source=initial_state_path,
            initial_source_hash=initial_state_path === nothing ? nothing :
                bytes2hex(SHA.sha256(read(initial_state_path))))
end

normalized_error(a, b) = maximum(abs, a .- b) / (1 + maximum(abs, b))
