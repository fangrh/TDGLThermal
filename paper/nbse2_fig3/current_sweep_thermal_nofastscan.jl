# TDGL + Thermal No-FastScan Current Sweep Script
# Usage:
#   julia current_sweep_thermal_nofastscan.jl config2.yaml
# Strategy:
# 1. Single-core: Fast scan ramp up (0→Jpeak) then ramp down (Jpeak→0), save states
# 2. Distributed: Each current point starts from fastscan state, runs stable+evolution
# 3. Save each current point's data to separate H5 file
#
# Usage:
#   julia -p 8 current_sweep_thermal_fastscan_standalone_V2.jl [config.yaml]

# ============================================================================
# USER CONFIGURABLE PARAMETERS - Modify these as needed
# ============================================================================

# --- Grid Parameters ---
const PARAM_Nx = 30             # Grid points in x direction
const PARAM_Ny = 20              # Grid points in y direction
const PARAM_hx = 0.5             # Grid spacing in x
const PARAM_hy = 0.5             # Grid spacing in y

# --- TDGL Physical Parameters ---
const PARAM_kappa = 4.0         # GL parameter κ
const PARAM_sigma = 1.0          # Normal conductivity σ (uniform, used if sigma1/2/3 not set)
const PARAM_sigma1 = nothing     # Normal conductivity for bottom region (nothing = use sigma)
const PARAM_sigma2 = nothing     # Normal conductivity for middle region (nothing = use sigma)
const PARAM_sigma3 = nothing     # Normal conductivity for top region (nothing = use sigma)
const PARAM_u = 5.79             # Relaxation parameter u
const PARAM_gamma = 10.0         # TDGL γ parameter

# --- Thermal Parameters ---
const PARAM_C_eff = 1.0          # Effective heat capacity
const PARAM_k_eff = 0.1          # Effective thermal conductivity
const PARAM_T0 = 0.0             # Bath temperature
const PARAM_eta_env = 10.0       # Heat transfer coefficient (environment)
const PARAM_eta_hole = 0.1       # Heat transfer coefficient (hole region)
const PARAM_holewidth = 10.0     # Width of the hole region (centered at x=0)

# --- Non-uniform Tc Parameters (three regions) ---
const PARAM_Tc1 = 1.0            # Tc for bottom region
const PARAM_Tc2 = 1.0            # Tc for middle region
const PARAM_Tc3 = 1.0            # Tc for top region
const PARAM_Tc1gap = 0.0         # Fraction of height with Tc1 at bottom (0 = no Tc1 region)
const PARAM_Tc3gap = 0.0         # Fraction of height with Tc3 at top (0 = no Tc3 region)

# --- External Fields ---
const PARAM_He = 0.0             # External magnetic field

# --- Boundary Conditions ---
const PARAM_bc_x = :dirichlet    # x-direction BC: :periodic, :neumann, or :dirichlet (psi=1, T=T0)
const PARAM_bc_x_induced_field_scale = 0.0  # x-boundary induced magnetic field scale; actual field = scale * Je
const PARAM_bc_y = :neumann      # y-direction BC: :neumann or :dirichlet (psi=0, T=T0)
const PARAM_bc_y_coverage = 1.0  # Electrode coverage (0-1): 1.0=full, 0.5=middle half, centered
const PARAM_loop = false         # If true, non-electrode y-boundaries form periodic loops
const PARAM_field_side = :both   # :both, :upper, or :lower - which side gets current-induced H field
const PARAM_psi_y_zero = false   # If true, psi=0 at all y boundaries (NS contact)
const PARAM_thermal_enabled = true  # If false, disable thermal coupling (T stays at T0)
const PARAM_epsilon_type = :linear  # :linear → ε=Tc-T, :bcs → ε=Tc/T-1

# --- Noise Parameters ---
const PARAM_noise_strength = 0.001  # Noise amplitude for ψ nucleation (0 = no noise)
                                    # Helps recovery from normal to superconducting state

# --- Current Sweep Parameters ---
const PARAM_Jpeak = 0.6          # Peak current density
const PARAM_n_current_points = 61 # Number of current points

# --- FastScan Parameters (single core) ---
const PARAM_rampup_time = 200.0   # Total ramp-up time
const PARAM_rampdown_time = 200.0 # Total ramp-down time

# --- Distributed Run Parameters ---
const PARAM_stable_time = 500.0   # Simulation time per current point
const PARAM_dt_snapshots = 1.0    # Time interval between snapshots (save every dt_snapshots)

# --- Data Processing Parameters ---
const PARAM_skip_ratio = 0.3     # Skip first 30% of data when computing averages

# --- Solver Parameters ---
const PARAM_abstol = 1e-4          # Absolute tolerance for ODE solver
const PARAM_reltol = 1e-4          # Relative tolerance for ODE solver

include(joinpath(@__DIR__, "animation_data_utils.jl"))

# ============================================================================
# END OF USER CONFIGURABLE PARAMETERS
# ============================================================================
const DEFAULT_LOCAL_N_WORKERS = 8  # Local fallback when not launched with Julia -p N

using Distributed

# Add workers if not already added
function initialize_workers(n::Int)
    if nworkers() == 1
        # Detect SLURM environment
        is_slurm = haskey(ENV, "SLURM_JOB_ID") && haskey(ENV, "SLURM_JOB_NAME")

        if is_slurm
            println("SLURM environment detected")
            println("Using Julia's built-in parallelization from SLURM process allocation")
            println("Workers will be managed by Julia's -p flag from submit.sh")
            # In SLURM, workers are already created by julia -p N
            # Don't use addprocs() - the workers are already available via nprocs()
            println("Running with $(nworkers()) workers (SLURM allocated)")
        else
            # Local execution - add workers via TCP
            n_add = min(n, Sys.CPU_THREADS - 1)
            println("Adding $n_add workers (requested: $n, available: $(Sys.CPU_THREADS))")
            addprocs(n_add)
            println("Running with $(nworkers()) workers")
        end
    else
        println("Using existing $(nworkers()) workers")
    end
    return nworkers()
end

# Initialize workers (only add workers if not in SLURM environment)
# Check both SLURM env vars AND if workers are actually allocated (> 1)
is_slurm_env = haskey(ENV, "SLURM_JOB_ID") && haskey(ENV, "SLURM_JOB_NAME")
if !is_slurm_env || nprocs() > 1
    # Already have workers (either from SLURM or previously added)
    println("Workers already available: $(nworkers())")
else
    initialize_workers(DEFAULT_LOCAL_N_WORKERS)
end

# ============================================================================
# All worker-required types and functions (inlined from TDGLFastScan module)
# ============================================================================

# Import Printf on main process (for functions outside @everywhere)
using Printf

@everywhere begin

include(joinpath(@__DIR__, "animation_data_utils.jl"))

using LinearAlgebra
using Statistics
using Printf  # Import full module for Printf.format()
using HDF5
using OrdinaryDiffEq

# ============================================================================
# Parameter Structure
# ============================================================================
mutable struct TDGLThermalParams
    Nx::Int
    Ny::Int
    hx::Float64
    hy::Float64
    dt::Float64
    kappa::Float64
    sigma1::Float64         # Normal conductivity for bottom region
    sigma2::Float64         # Normal conductivity for middle region
    sigma3::Float64         # Normal conductivity for top region
    u::Float64
    gamma::Float64
    C_eff::Float64
    k_eff::Float64
    T0::Float64
    eta_env::Float64
    eta_hole::Float64
    holewidth::Float64
    Je::Float64
    He::Float64
    bc_x::Symbol
    bc_x_induced_field_scale::Float64
    bc_y::Symbol
    bc_y_coverage::Float64  # Fraction of x-width covered by electrode (0-1), centered
    loop::Bool              # If true, non-electrode y-boundaries form periodic loops
    field_side::Symbol      # :both, :upper, or :lower - which side gets current-induced field
    Tc1::Float64           # Tc for bottom region
    Tc2::Float64           # Tc for middle region
    Tc3::Float64           # Tc for top region
    Tc1gap::Float64        # Fraction of height with Tc1 at bottom (0 = no Tc1 region)
    Tc3gap::Float64        # Fraction of height with Tc3 at top (0 = no Tc3 region)
    psi_y_zero::Bool        # If true, psi=0 at all y boundaries (NS contact)
    thermal_enabled::Bool   # If false, disable thermal coupling (T stays at T0)
    epsilon_type::Symbol    # :linear → ε=Tc-T, :bcs → ε=Tc/T-1
    noise_strength::Float64 # Noise amplitude for ψ nucleation
end

function TDGLThermalParams(;
    Nx=100, Ny=100, hx=0.5, hy=0.5, dt=0.0,
    kappa=10.0, sigma=1.0, sigma1=nothing, sigma2=nothing, sigma3=nothing,
    u=5.79, gamma=10.0,
    C_eff=1.0, k_eff=0.1, T0=0.0,
    eta_env=10.0, eta_hole=0.1, holewidth=-1.0,
    Je=0.0, He=0.0, bc_x=:periodic, bc_x_induced_field_scale=0.0, bc_y=:neumann, bc_y_coverage=1.0, loop=false,
    field_side=:both,
    Tc1=1.0, Tc2=1.0, Tc3=1.0, Tc1gap=0.0, Tc3gap=0.0,
    psi_y_zero=false,  thermal_enabled=true,  epsilon_type=:linear,  noise_strength=0.0)
    if dt <= 0.0
        dt = 0.1 * min(hx^2, hy^2) / 4.0
    end
    if holewidth < 0
        holewidth = Nx * hx / 3.0
    end
    # If sigma1/2/3 not specified, use uniform sigma
    s1 = sigma1 === nothing ? sigma : sigma1
    s2 = sigma2 === nothing ? sigma : sigma2
    s3 = sigma3 === nothing ? sigma : sigma3
    if bc_x == :periodic && bc_x_induced_field_scale != 0.0
        throw(ArgumentError("boundary.bc_x_induced_field_scale requires non-periodic bc_x"))
    end
    return TDGLThermalParams(Nx, Ny, hx, hy, dt, kappa, s1, s2, s3, u, gamma,
                             C_eff, k_eff, T0, eta_env, eta_hole, holewidth,
                             Je, He, bc_x, bc_x_induced_field_scale, bc_y, bc_y_coverage, loop, field_side,
                             Tc1, Tc2, Tc3, Tc1gap, Tc3gap, psi_y_zero, thermal_enabled,
                             Symbol(epsilon_type), noise_strength)
end

function with_current(p::TDGLThermalParams, Je::Float64)
    return TDGLThermalParams(p.Nx, p.Ny, p.hx, p.hy, p.dt,
                             p.kappa, p.sigma1, p.sigma2, p.sigma3, p.u, p.gamma,
                             p.C_eff, p.k_eff, p.T0, p.eta_env, p.eta_hole, p.holewidth,
                             Je, p.He, p.bc_x, p.bc_x_induced_field_scale, p.bc_y, p.bc_y_coverage, p.loop, p.field_side,
                             p.Tc1, p.Tc2, p.Tc3, p.Tc1gap, p.Tc3gap, p.psi_y_zero, p.thermal_enabled,
                             p.epsilon_type, p.noise_strength)
end

# ============================================================================
# State Structure
# ============================================================================
mutable struct TDGLThermalState
    psi::Matrix{ComplexF64}
    Ax::Matrix{Float64}
    Ay::Matrix{Float64}
    T::Matrix{Float64}
    Ux::Matrix{ComplexF64}
    Uy::Matrix{ComplexF64}
end

# ============================================================================
# Sweep Configuration
# ============================================================================
mutable struct SweepConfig
    Jpeak::Float64
    n_current_points::Int
    rampup_time::Float64      # Keep for now, will remove later
    rampdown_time::Float64     # Keep for now, will remove later
    stable_time::Float64
    dt_snapshots::Float64
    skip_ratio::Float64
    run_up::Bool
    run_down::Bool
    generate_mp4::Bool
    mp4_fps::Int
    frames_per_point::Int
    # NEW: Initial conditions
    psi_up_init::Float64
    T_initial_up::Float64
    psi_down_init::Float64
    T_initial_down::Float64
    input_folder::Union{Nothing, String}
end

function initialize_state_clean(p::TDGLThermalParams)
    # Initialize psi based on Tc distribution: |ψ|² = ε
    psi = zeros(ComplexF64, p.Nx+1, p.Ny+1)
    Ax = zeros(Float64, p.Nx, p.Ny+1)
    Ay = zeros(Float64, p.Nx+1, p.Ny)
    T = fill(p.T0, (p.Nx+1, p.Ny+1))

    # Compute Tc distribution and set psi accordingly
    Ly = p.Ny * p.hy
    y_lower = Ly * p.Tc1gap
    y_upper = Ly * (1.0 - p.Tc3gap)

    for j in 1:p.Ny+1
        y = (j - 1) * p.hy
        if y < y_lower
            Tc_val = p.Tc1
        elseif y >= y_upper
            Tc_val = p.Tc3
        else
            Tc_val = p.Tc2
        end

        # Compute ε depending on epsilon_type
        if Tc_val > 0
            if p.epsilon_type == :bcs
                # ε = Tc/T - 1
                epsilon = p.T0 > 0 ? Tc_val / p.T0 - 1.0 : 1.0
            else
                # ε = Tc - T (linear, default)
                epsilon = Tc_val - p.T0
            end
            psi_mag = epsilon > 0 ? sqrt(min(epsilon, 1.0)) : 0.0
        else
            psi_mag = 0.0
        end

        for i in 1:p.Nx+1
            psi[i, j] = psi_mag
        end
    end

    Ux = exp.(-im * p.hx .* Ax)
    Uy = exp.(-im * p.hy .* Ay)
    return TDGLThermalState(psi, Ax, Ay, T, Ux, Uy)
end

function initialize_state_with_params(p::TDGLThermalParams, psi_init::Float64, T_init::Float64)
    # Initialize state with uniform psi and T values
    psi = fill(ComplexF64(psi_init * (psi_init >= 0 ? 1.0 : im)), p.Nx+1, p.Ny+1)
    Ax = zeros(Float64, p.Nx, p.Ny+1)
    Ay = zeros(Float64, p.Nx+1, p.Ny)
    T = fill(T_init, (p.Nx+1, p.Ny+1))

    Ux = exp.(-im * p.hx .* Ax)
    Uy = exp.(-im * p.hy .* Ay)
    return TDGLThermalState(psi, Ax, Ay, T, Ux, Uy)
end

function initialize_state_from_fields(p::TDGLThermalParams, psi::Matrix{ComplexF64},
                                      Ax::Matrix{Float64}, Ay::Matrix{Float64}, T::Matrix{Float64})
    expected_psi_size = (p.Nx + 1, p.Ny + 1)
    expected_Ax_size = (p.Nx, p.Ny + 1)
    expected_Ay_size = (p.Nx + 1, p.Ny)

    size(psi) == expected_psi_size || error("Continuation psi size mismatch: expected $expected_psi_size, got $(size(psi))")
    size(T) == expected_psi_size || error("Continuation T size mismatch: expected $expected_psi_size, got $(size(T))")
    size(Ax) == expected_Ax_size || error("Continuation Ax size mismatch: expected $expected_Ax_size, got $(size(Ax))")
    size(Ay) == expected_Ay_size || error("Continuation Ay size mismatch: expected $expected_Ay_size, got $(size(Ay))")

    Ux = exp.(-im * p.hx .* Ax)
    Uy = exp.(-im * p.hy .* Ay)
    return TDGLThermalState(copy(psi), copy(Ax), copy(Ay), copy(T), Ux, Uy)
end

# ============================================================================
# Distributions
# ============================================================================
function compute_eta_distribution(p::TDGLThermalParams)
    eta = zeros(Float64, p.Nx+1, p.Ny+1)
    Lx = p.Nx * p.hx
    half_hole = p.holewidth / 2.0
    for j in 1:p.Ny+1
        for i in 1:p.Nx+1
            x = (i - 1) * p.hx - Lx / 2.0
            eta[i, j] = abs(x) < half_hole ? p.eta_hole : p.eta_env
        end
    end
    return eta
end

function compute_Tc_distribution(p::TDGLThermalParams)
    # Three regions: bottom (Tc1), middle (Tc2), top (Tc3)
    # Tc1gap: fraction of height for Tc1 at bottom (y < y_lower)
    # Tc3gap: fraction of height for Tc3 at top (y > y_upper)
    # Middle region (y_lower <= y < y_upper) has Tc2
    Tc = zeros(Float64, p.Nx+1, p.Ny+1)
    Ly = p.Ny * p.hy
    y_lower = Ly * p.Tc1gap
    y_upper = Ly * (1.0 - p.Tc3gap)

    for j in 1:p.Ny+1
        y = (j - 1) * p.hy
        if y < y_lower
            Tc_val = p.Tc1
        elseif y >= y_upper
            Tc_val = p.Tc3
        else
            Tc_val = p.Tc2
        end
        for i in 1:p.Nx+1
            Tc[i, j] = Tc_val
        end
    end
    return Tc
end

function compute_sigma_distribution(p::TDGLThermalParams)
    # Three regions: bottom (sigma1), middle (sigma2), top (sigma3)
    # Uses same gaps as Tc distribution
    sigma = zeros(Float64, p.Nx+1, p.Ny+1)
    Ly = p.Ny * p.hy
    y_lower = Ly * p.Tc1gap
    y_upper = Ly * (1.0 - p.Tc3gap)

    for j in 1:p.Ny+1
        y = (j - 1) * p.hy
        if y < y_lower
            sigma_val = p.sigma1
        elseif y >= y_upper
            sigma_val = p.sigma3
        else
            sigma_val = p.sigma2
        end
        for i in 1:p.Nx+1
            sigma[i, j] = sigma_val
        end
    end
    return sigma
end

# ============================================================================
# Boundary & Link Variables
# ============================================================================
function update_link_variables!(state::TDGLThermalState, p::TDGLThermalParams)
    @. state.Ux = exp(-im * p.hx * state.Ax)
    @. state.Uy = exp(-im * p.hy * state.Ay)
end

function apply_boundary_conditions!(state::TDGLThermalState, p::TDGLThermalParams)
    Lx = p.Nx * p.hx
    Ly = p.Ny * p.hy

    # Calculate electrode coverage region (centered)
    coverage = clamp(p.bc_y_coverage, 0.0, 1.0)
    margin = (1.0 - coverage) / 2.0
    x_left = -Lx/2 + margin * Lx
    x_right = Lx/2 - margin * Lx

    # Induced magnetic field from current
    # field_side controls which boundary gets current-induced field:
    #   :both  - both boundaries get ±H_induced, both follow bc_y_coverage
    #   :upper - only upper boundary gets field (follows bc_y_coverage), lower is zero (full boundary)
    #   :lower - only lower boundary gets field (follows bc_y_coverage), upper is zero (full boundary)
    H_induced = p.Je * Ly / (2 * p.kappa^2)

    if p.field_side == :both
        H_upper_electrode = -H_induced + p.He
        H_lower_electrode = H_induced + p.He
    elseif p.field_side == :upper
        # Only upper boundary gets current-induced field, lower is zero (just He)
        H_upper_electrode = -2 * H_induced + p.He
        H_lower_electrode = p.He
    elseif p.field_side == :lower
        # Only lower boundary gets current-induced field, upper is zero (just He)
        H_upper_electrode = p.He
        H_lower_electrode = 2 * H_induced + p.He
    else
        # Default to symmetric
        H_upper_electrode = -H_induced + p.He
        H_lower_electrode = H_induced + p.He
    end

    for i in 1:p.Nx
        # x position at this grid point (Ax is on edges, use midpoint)
        x = (i - 0.5) * p.hx - Lx / 2.0
        in_electrode = (x >= x_left) && (x <= x_right)

        # Apply boundary conditions for lower boundary (j=1)
        if p.field_side == :upper
            # Zero-field side: Neumann BC (∂Ax/∂y = 0)
            state.Ax[i, 1] = state.Ax[i, 2]
        elseif p.field_side == :lower
            # Field side: follows bc_y_coverage
            if in_electrode
                state.Ax[i, 1] = state.Ax[i, 2] - p.hy * H_lower_electrode
            elseif p.loop
                state.Ax[i, 1] = state.Ax[i, p.Ny]
            else
                state.Ax[i, 1] = state.Ax[i, 2] - p.hy * p.He
            end
        else # :both
            if in_electrode
                state.Ax[i, 1] = state.Ax[i, 2] - p.hy * H_lower_electrode
            elseif p.loop
                state.Ax[i, 1] = state.Ax[i, p.Ny]
            else
                state.Ax[i, 1] = state.Ax[i, 2] - p.hy * p.He
            end
        end

        # Apply boundary conditions for upper boundary (j=Ny+1)
        if p.field_side == :lower
            # Zero-field side: Neumann BC (∂Ax/∂y = 0)
            state.Ax[i, p.Ny+1] = state.Ax[i, p.Ny]
        elseif p.field_side == :upper
            # Field side: follows bc_y_coverage
            if in_electrode
                state.Ax[i, p.Ny+1] = state.Ax[i, p.Ny] + p.hy * H_upper_electrode
            elseif p.loop
                state.Ax[i, p.Ny+1] = state.Ax[i, 2]
            else
                state.Ax[i, p.Ny+1] = state.Ax[i, p.Ny] + p.hy * p.He
            end
        else # :both
            if in_electrode
                state.Ax[i, p.Ny+1] = state.Ax[i, p.Ny] + p.hy * H_upper_electrode
            elseif p.loop
                state.Ax[i, p.Ny+1] = state.Ax[i, 2]
            else
                state.Ax[i, p.Ny+1] = state.Ax[i, p.Ny] + p.hy * p.He
            end
        end
    end

    for j in 1:p.Ny
        if p.bc_x == :neumann
            Hx_induced = p.bc_x_induced_field_scale * p.Je
            state.Ay[1, j] = state.Ay[2, j] - p.hx * Hx_induced
            state.Ay[p.Nx+1, j] = state.Ay[p.Nx, j] + p.hx * Hx_induced
        elseif p.bc_x == :dirichlet
            Hx_induced = p.bc_x_induced_field_scale * p.Je
            state.Ay[1, j] = state.Ay[2, j] - p.hx * Hx_induced
            state.Ay[p.Nx+1, j] = state.Ay[p.Nx, j] + p.hx * Hx_induced
        elseif p.bc_x == :periodic
            state.Ay[1, j] = state.Ay[p.Nx, j]
            state.Ay[p.Nx+1, j] = state.Ay[2, j]
        end
    end

    for j in 1:p.Ny+1
        if p.bc_x == :neumann
            # Covariant Neumann: (∇-iA)·n ψ = 0 (Prawitasari 2019, Eq. 7)
            # Left boundary: psi[1,j] = Ux*[1,j] * psi[2,j]
            # Right boundary: psi[Nx+1,j] = Ux[Nx,j] * psi[Nx,j]
            state.psi[1, j] = conj(state.Ux[1, j]) * state.psi[2, j]
            state.psi[p.Nx+1, j] = state.Ux[p.Nx, j] * state.psi[p.Nx, j]
            state.T[1, j] = state.T[2, j]
            state.T[p.Nx+1, j] = state.T[p.Nx, j]
        elseif p.bc_x == :dirichlet
            # Fixed superconducting state at left/right boundaries
            state.psi[1, j] = 1.0
            state.psi[p.Nx+1, j] = 1.0
            state.T[1, j] = p.T0
            state.T[p.Nx+1, j] = p.T0
        elseif p.bc_x == :periodic
            state.psi[1, j] = state.psi[p.Nx, j]
            state.psi[p.Nx+1, j] = state.psi[2, j]
            state.T[1, j] = state.T[p.Nx, j]
            state.T[p.Nx+1, j] = state.T[2, j]
        end
    end

    # Y-boundaries: ψ and T (uses same x_left, x_right from above)
    # Covariant Neumann: (∇-iA)·n ψ = 0 (Prawitasari 2019, Eq. 7)
    # Lower boundary: psi[i,1] = Uy*[i,1] * psi[i,2]
    # Upper boundary: psi[i,Ny+1] = Uy[i,Ny] * psi[i,Ny]
    for i in 1:p.Nx+1
        # Calculate x position for this grid point
        x = (i - 1) * p.hx - Lx / 2.0

        # Check if this point is under the electrode (based on bc_y_coverage)
        in_electrode = (x >= x_left) && (x <= x_right)

        # Get Uy values for covariant Neumann (handle boundary indices)
        Uy_lower = (i <= p.Nx+1 && 1 <= p.Ny) ? state.Uy[i, 1] : exp(-im * p.hy * 0.0)
        Uy_upper = (i <= p.Nx+1 && p.Ny >= 1) ? state.Uy[i, p.Ny] : exp(-im * p.hy * 0.0)

        # Lower boundary (j=1)
        if p.field_side == :upper
            # Zero-field side: always covariant Neumann BC (full boundary)
            state.T[i, 1] = state.T[i, 2]
            state.psi[i, 1] = conj(Uy_lower) * state.psi[i, 2]
        elseif p.field_side == :lower
            # Field side: follows bc_y_coverage and bc_y
            if in_electrode
                state.T[i, 1] = state.T[i, 2]
                if p.bc_y == :neumann
                    state.psi[i, 1] = conj(Uy_lower) * state.psi[i, 2]
                elseif p.bc_y == :dirichlet
                    state.psi[i, 1] = 0.0
                end
            elseif p.loop
                state.psi[i, 1] = state.psi[i, p.Ny]
                state.T[i, 1] = state.T[i, p.Ny]
            else
                state.T[i, 1] = state.T[i, 2]
                state.psi[i, 1] = conj(Uy_lower) * state.psi[i, 2]
            end
        else # :both
            if in_electrode
                state.T[i, 1] = state.T[i, 2]
                if p.bc_y == :neumann
                    state.psi[i, 1] = conj(Uy_lower) * state.psi[i, 2]
                elseif p.bc_y == :dirichlet
                    state.psi[i, 1] = 0.0
                end
            elseif p.loop
                state.psi[i, 1] = state.psi[i, p.Ny]
                state.T[i, 1] = state.T[i, p.Ny]
            else
                state.T[i, 1] = state.T[i, 2]
                state.psi[i, 1] = conj(Uy_lower) * state.psi[i, 2]
            end
        end

        # Upper boundary (j=Ny+1)
        if p.field_side == :lower
            # Zero-field side: always covariant Neumann BC (full boundary)
            state.T[i, p.Ny+1] = state.T[i, p.Ny]
            state.psi[i, p.Ny+1] = Uy_upper * state.psi[i, p.Ny]
        elseif p.field_side == :upper
            # Field side: follows bc_y_coverage and bc_y
            if in_electrode
                state.T[i, p.Ny+1] = state.T[i, p.Ny]
                if p.bc_y == :neumann
                    state.psi[i, p.Ny+1] = Uy_upper * state.psi[i, p.Ny]
                elseif p.bc_y == :dirichlet
                    state.psi[i, p.Ny+1] = 0.0
                end
            elseif p.loop
                state.psi[i, p.Ny+1] = state.psi[i, 2]
                state.T[i, p.Ny+1] = state.T[i, 2]
            else
                state.T[i, p.Ny+1] = state.T[i, p.Ny]
                state.psi[i, p.Ny+1] = Uy_upper * state.psi[i, p.Ny]
            end
        else # :both
            if in_electrode
                state.T[i, p.Ny+1] = state.T[i, p.Ny]
                if p.bc_y == :neumann
                    state.psi[i, p.Ny+1] = Uy_upper * state.psi[i, p.Ny]
                elseif p.bc_y == :dirichlet
                    state.psi[i, p.Ny+1] = 0.0
                end
            elseif p.loop
                state.psi[i, p.Ny+1] = state.psi[i, 2]
                state.T[i, p.Ny+1] = state.T[i, 2]
            else
                state.T[i, p.Ny+1] = state.T[i, p.Ny]
                state.psi[i, p.Ny+1] = Uy_upper * state.psi[i, p.Ny]
            end
        end
    end

    # Override: if psi_y_zero is true, force psi=0 at all y boundaries
    if p.psi_y_zero
        for i in 1:p.Nx+1
            state.psi[i, 1] = 0.0
            state.psi[i, p.Ny+1] = 0.0
        end
    end
end

# ============================================================================
# Physics Computations
# ============================================================================
function compute_dpsi_dt!(dpsi::Matrix{ComplexF64}, RHS::Matrix{ComplexF64},
                          state::TDGLThermalState, p::TDGLThermalParams,
                          Tc::Matrix{Float64})
    psi = state.psi
    Ux = state.Ux
    Uy = state.Uy
    T = state.T
    hx2 = p.hx^2
    hy2 = p.hy^2
    gamma2 = p.gamma^2
    use_bcs = (p.epsilon_type == :bcs)
    noise_amp = p.noise_strength

    for j in 2:p.Ny
        @simd for i in 2:p.Nx
            # Covariant Laplacian using link variables
            laplacian_term = (conj(Ux[i-1,j]) * psi[i-1,j] - 2*psi[i,j] + Ux[i,j] * psi[i+1,j]) / hx2 +
                            (conj(Uy[i,j-1]) * psi[i,j-1] - 2*psi[i,j] + Uy[i,j] * psi[i,j+1]) / hy2

            # GL potential: nonlinear term = (ε - |ψ|²)ψ
            psi_abs2 = abs2(psi[i,j])
            if use_bcs
                # ε = Tc/T - 1
                epsilon_T = T[i,j] > 0 ? Tc[i,j] / T[i,j] - 1.0 : 1.0
            else
                # ε = Tc - T (linear, default)
                epsilon_T = Tc[i,j] - T[i,j]
            end
            nonlinear_term = (epsilon_T - psi_abs2) * psi[i,j]
            RHS[i,j] = laplacian_term + nonlinear_term

            if gamma2 != 0.0
                # Kramer-Watts-Tobin TDGL:
                # u/sqrt(1+γ²|ψ|²) * (∂ψ/∂t + γ²/2 * ∂|ψ|²/∂t * ψ) = RHS
                # Solved for ∂ψ/∂t:
                alpha = p.u / sqrt(1 + gamma2 * psi_abs2)
                RHS_over_alpha = RHS[i,j] / alpha
                c = real(conj(psi[i,j]) * RHS_over_alpha) / (1 + gamma2 * psi_abs2)
                dpsi[i,j] = RHS_over_alpha - gamma2 * c * psi[i,j]
            else
                # Simple TDGL: u * ∂ψ/∂t = RHS
                dpsi[i,j] = RHS[i,j] / p.u
            end

            # Add Langevin noise for nucleation (helps recovery from normal state)
            # Noise is scaled by sqrt(ε) to be stronger when ψ is small
            if noise_amp > 0.0
                # Gaussian white noise with complex amplitude
                noise_scale = noise_amp * sqrt(max(epsilon_T, 0.0) + 0.1)
                dpsi[i,j] += noise_scale * (randn() + im * randn())
            end
        end
    end
end

function compute_supercurrent!(Jsx::Matrix{Float64}, Jsy::Matrix{Float64},
                               state::TDGLThermalState, p::TDGLThermalParams)
    psi = state.psi
    Ux = state.Ux
    Uy = state.Uy

    for j in 2:p.Ny
        @simd for i in 1:p.Nx
            Jsx[i,j] = imag(conj(psi[i,j]) * Ux[i,j] * psi[i+1,j]) / p.hx
        end
    end
    for j in 1:p.Ny
        @simd for i in 2:p.Nx
            Jsy[i,j] = imag(conj(psi[i,j]) * Uy[i,j] * psi[i,j+1]) / p.hy
        end
    end
end

function compute_dA_dt!(dAx::Matrix{Float64}, dAy::Matrix{Float64},
                        state::TDGLThermalState, p::TDGLThermalParams,
                        Jsx::Matrix{Float64}, Jsy::Matrix{Float64},
                        sigma::Matrix{Float64})
    Ax = state.Ax
    Ay = state.Ay
    kappa2 = p.kappa^2
    hx2 = p.hx^2
    hy2 = p.hy^2

    fill!(dAx, 0.0)
    fill!(dAy, 0.0)

    for j in 2:p.Ny
        @simd for i in 2:p.Nx-1
            d2Ax_dy2 = (Ax[i,j+1] - 2*Ax[i,j] + Ax[i,j-1]) / hy2
            # Ax is on (i, j) edges, use sigma at nearest node
            sigma_local = (sigma[i,j] + sigma[i+1,j]) / 2
            dAx[i,j] = (Jsx[i,j] + kappa2 * d2Ax_dy2) / sigma_local
        end
    end
    for j in 2:p.Ny-1
        @simd for i in 2:p.Nx
            d2Ay_dx2 = (Ay[i+1,j] - 2*Ay[i,j] + Ay[i-1,j]) / hx2
            # Ay is on (i, j) edges, use sigma at nearest node
            sigma_local = (sigma[i,j] + sigma[i,j+1]) / 2
            dAy[i,j] = (Jsy[i,j] + kappa2 * d2Ay_dx2) / sigma_local
        end
    end
end

function compute_heat_production!(W_total::Matrix{Float64},
                                  state::TDGLThermalState, p::TDGLThermalParams,
                                  dpsi::Matrix{ComplexF64},
                                  dAx::Matrix{Float64}, dAy::Matrix{Float64},
                                  sigma::Matrix{Float64})
    psi = state.psi
    gamma2 = p.gamma^2
    fill!(W_total, 0.0)

    for j in 2:p.Ny
        @simd for i in 2:p.Nx
            dAx_avg = (i > 1 && i <= p.Nx) ? (dAx[i,j] + dAx[i-1,j]) / 2 : 0.0
            dAy_avg = (j > 1 && j <= p.Ny) ? (dAy[i,j] + dAy[i,j-1]) / 2 : 0.0
            E_sq = dAx_avg^2 + dAy_avg^2
            W_joule = 2 * sigma[i,j] * E_sq

            psi_abs2 = abs2(psi[i,j])
            dpsi_abs2 = abs2(dpsi[i,j])
            d_psi_abs2_dt = 2 * real(conj(psi[i,j]) * dpsi[i,j])
            prefactor = 2 * p.u / sqrt(1 + gamma2 * psi_abs2)
            W_GL = prefactor * (dpsi_abs2 + gamma2 / 4 * d_psi_abs2_dt^2)

            W_total[i,j] = W_joule + W_GL
        end
    end
end

function compute_dT_dt!(dT::Matrix{Float64}, state::TDGLThermalState,
                        p::TDGLThermalParams, W_total::Matrix{Float64},
                        eta::Matrix{Float64})
    T = state.T
    hx2 = p.hx^2
    hy2 = p.hy^2
    fill!(dT, 0.0)

    for j in 2:p.Ny
        @simd for i in 2:p.Nx
            laplacian_T = (T[i-1,j] - 2*T[i,j] + T[i+1,j]) / hx2 +
                          (T[i,j-1] - 2*T[i,j] + T[i,j+1]) / hy2
            diffusion = p.k_eff * laplacian_T
            source = 0.5 * W_total[i,j]
            cooling = -eta[i,j] * (T[i,j] - p.T0)
            dT[i,j] = (diffusion + source + cooling) / p.C_eff
        end
    end
end

function compute_average_order_parameter(state::TDGLThermalState)
    return mean(abs.(state.psi[2:end-1, 2:end-1]))
end

function compute_average_temperature(state::TDGLThermalState)
    return mean(state.T[2:end-1, 2:end-1])
end

function compute_voltage_from_du(du, p::TDGLThermalParams)
    len_psi = (p.Nx+1) * (p.Ny+1)
    len_Ax = p.Nx * (p.Ny+1)
    dAx = reshape(du[len_psi+1 : len_psi+len_Ax], (p.Nx, p.Ny+1))
    Lx = p.Nx * p.hx
    V = -mean(dAx[2:end-1, 2:end-1]) * Lx
    return V
end

function compute_voltage_hole_from_du(du, p::TDGLThermalParams)
    len_psi = (p.Nx+1) * (p.Ny+1)
    len_Ax = p.Nx * (p.Ny+1)
    dAx = reshape(du[len_psi+1 : len_psi+len_Ax], (p.Nx, p.Ny+1))
    Lx = p.Nx * p.hx
    half_hole = p.holewidth / 2.0

    # Collect dAx values only in the hole region (|x| < holewidth/2)
    sum_val = 0.0
    count = 0
    for j in 2:p.Ny
        for i in 2:p.Nx-1
            # Ax edge midpoint x-position
            x = (i - 0.5) * p.hx - Lx / 2.0
            if abs(x) < half_hole
                sum_val += dAx[i, j]
                count += 1
            end
        end
    end
    V_hole = count > 0 ? -sum_val / count * Lx : 0.0
    return V_hole
end

# ============================================================================
# ODE Cache
# ============================================================================
struct TDGLThermalCache
    params::TDGLThermalParams
    state::TDGLThermalState
    dpsi::Matrix{ComplexF64}
    RHS_psi::Matrix{ComplexF64}
    dAx::Matrix{Float64}
    dAy::Matrix{Float64}
    dT::Matrix{Float64}
    Jsx::Matrix{Float64}
    Jsy::Matrix{Float64}
    W_total::Matrix{Float64}
    eta::Matrix{Float64}
    Tc::Matrix{Float64}
    sigma::Matrix{Float64}
end

function TDGLThermalCache(p::TDGLThermalParams)
    state = initialize_state_clean(p)
    dpsi = zeros(ComplexF64, p.Nx+1, p.Ny+1)
    RHS_psi = zeros(ComplexF64, p.Nx+1, p.Ny+1)
    dAx = zeros(Float64, p.Nx, p.Ny+1)
    dAy = zeros(Float64, p.Nx+1, p.Ny)
    dT = zeros(Float64, p.Nx+1, p.Ny+1)
    Jsx = zeros(Float64, p.Nx, p.Ny+1)
    Jsy = zeros(Float64, p.Nx+1, p.Ny)
    W_total = zeros(Float64, p.Nx+1, p.Ny+1)
    eta = compute_eta_distribution(p)
    Tc = compute_Tc_distribution(p)
    sigma = compute_sigma_distribution(p)
    return TDGLThermalCache(p, state, dpsi, RHS_psi, dAx, dAy, dT, Jsx, Jsy, W_total, eta, Tc, sigma)
end

# ============================================================================
# State Vector Conversion
# ============================================================================
function state_to_vector(state::TDGLThermalState)
    return vcat(vec(state.psi), vec(state.Ax), vec(state.Ay), vec(state.T))
end

function vector_to_state!(state::TDGLThermalState, u, p::TDGLThermalParams)
    nx, ny = p.Nx, p.Ny
    len_psi = (nx+1) * (ny+1)
    len_Ax = nx * (ny+1)
    len_Ay = (nx+1) * ny
    state.psi .= reshape(u[1:len_psi], (nx+1, ny+1))
    state.Ax .= reshape(real.(u[len_psi+1 : len_psi+len_Ax]), (nx, ny+1))
    state.Ay .= reshape(real.(u[len_psi+len_Ax+1 : len_psi+len_Ax+len_Ay]), (nx+1, ny))
    state.T .= reshape(real.(u[len_psi+len_Ax+len_Ay+1 : end]), (nx+1, ny+1))
    update_link_variables!(state, p)
end

# ============================================================================
# ODE RHS Function
# ============================================================================
function fdm_rhs_thermal!(du, u, cache::TDGLThermalCache, t)
    (; params, state, dpsi, RHS_psi, dAx, dAy, dT, Jsx, Jsy, W_total, eta, Tc, sigma) = cache

    vector_to_state!(state, u, params)
    apply_boundary_conditions!(state, params)
    compute_dpsi_dt!(dpsi, RHS_psi, state, params, Tc)
    compute_supercurrent!(Jsx, Jsy, state, params)
    compute_dA_dt!(dAx, dAy, state, params, Jsx, Jsy, sigma)

    # Only compute thermal evolution if enabled
    if params.thermal_enabled
        compute_heat_production!(W_total, state, params, dpsi, dAx, dAy, sigma)
        compute_dT_dt!(dT, state, params, W_total, eta)
    else
        # Keep temperature constant at T0
        dT .= 0.0
    end

    nx, ny = params.Nx, params.Ny
    len_psi = (nx+1) * (ny+1)
    len_Ax = nx * (ny+1)
    len_Ay = (nx+1) * ny
    len_psi_plus_Ax = len_psi + len_Ax

    du[1:len_psi] .= vec(dpsi)
    du[len_psi+1:len_psi_plus_Ax] .= vec(dAx)
    du[len_psi_plus_Ax+1:len_psi_plus_Ax+len_Ay] .= vec(dAy)
    du[len_psi_plus_Ax+len_Ay+1:end] .= vec(dT)
end

# ============================================================================
# FastScan Function
# ============================================================================
function run_fastscan(p::TDGLThermalParams, J_values::Vector{Float64},
                      rampup_time::Float64, rampdown_time::Float64;
                      verbose::Bool=true)
    n_points = length(J_values)
    dt_rampup = rampup_time / max(n_points - 1, 1)
    dt_rampdown = rampdown_time / max(n_points - 1, 1)

    fastscan_up = Vector{Vector{ComplexF64}}(undef, n_points)
    fastscan_down = Vector{Vector{ComplexF64}}(undef, n_points)

    state = initialize_state_clean(p)
    u0 = state_to_vector(state)

    # Ramp Up
    if verbose
        println("[FastScan] Ramp Up: 0 → $(J_values[end])")
    end

    for (idx, Je) in enumerate(J_values)
        p_current = with_current(p, Je)
        cache = TDGLThermalCache(p_current)
        vector_to_state!(cache.state, u0, p_current)

        if idx > 1
            tspan = (0.0, dt_rampup)
            prob = ODEProblem(fdm_rhs_thermal!, u0, tspan, cache)
            sol = solve(prob, Tsit5(), abstol=1e-5, reltol=1e-5)
            u0 = sol.u[end]
        end

        fastscan_up[idx] = copy(u0)

        if verbose
            du_check = similar(u0)
            fdm_rhs_thermal!(du_check, u0, cache, 0.0)
            V_check = compute_voltage_from_du(du_check, p_current)
            vector_to_state!(cache.state, u0, p_current)
            psi_avg = compute_average_order_parameter(cache.state)
            T_avg = compute_average_temperature(cache.state)
            msg = Printf.format(Printf.Format("  [UP] %2d/%2d Je=%.4f: V=%.4e, <|ψ|>=%.4f, <T>=%.4f\n"),
                               idx, n_points, Je, V_check, psi_avg, T_avg)
            write(stdout, msg)
        end
    end

    # Ramp Down
    if verbose
        println("[FastScan] Ramp Down: $(J_values[end]) → 0")
    end

    u0 = copy(fastscan_up[end])

    for (idx, Je) in enumerate(reverse(J_values))
        store_idx = n_points - idx + 1
        p_current = with_current(p, Je)
        cache = TDGLThermalCache(p_current)
        vector_to_state!(cache.state, u0, p_current)

        if idx > 1
            tspan = (0.0, dt_rampdown)
            prob = ODEProblem(fdm_rhs_thermal!, u0, tspan, cache)
            sol = solve(prob, Tsit5(), abstol=1e-5, reltol=1e-5)
            u0 = sol.u[end]
        end

        fastscan_down[store_idx] = copy(u0)

        if verbose
            du_check = similar(u0)
            fdm_rhs_thermal!(du_check, u0, cache, 0.0)
            V_check = compute_voltage_from_du(du_check, p_current)
            vector_to_state!(cache.state, u0, p_current)
            psi_avg = compute_average_order_parameter(cache.state)
            T_avg = compute_average_temperature(cache.state)
            msg = Printf.format(Printf.Format("  [DN] %2d/%2d Je=%.4f: V=%.4e, <|ψ|>=%.4f, <T>=%.4f\n"),
                               idx, n_points, Je, V_check, psi_avg, T_avg)
            write(stdout, msg)
        end
    end

    return fastscan_up, fastscan_down
end

# ============================================================================
# FastScan Pipeline Function (with callback for streaming dispatch)
# ============================================================================
"""
    run_fastscan_pipeline(p, J_values, rampup_time, rampdown_time, on_state_ready; verbose=true)

Run fastscan with callback for pipeline processing.
`on_state_ready(idx, Je, u0, phase)` is called each time a state is ready.
- phase: :up or :down
- idx: index in J_values (1 to n_points)
Returns fastscan_up, fastscan_down arrays for reference.
"""
function run_fastscan_pipeline(p::TDGLThermalParams, J_values::Vector{Float64},
                               rampup_time::Float64, rampdown_time::Float64,
                               on_state_ready::Function;
                               verbose::Bool=true)
    n_points = length(J_values)
    dt_rampup = rampup_time / max(n_points - 1, 1)
    dt_rampdown = rampdown_time / max(n_points - 1, 1)

    fastscan_up = Vector{Vector{ComplexF64}}(undef, n_points)
    fastscan_down = Vector{Vector{ComplexF64}}(undef, n_points)

    state = initialize_state_clean(p)
    u0 = state_to_vector(state)

    # Ramp Up
    if verbose
        println("[FastScan Pipeline] Ramp Up: 0 → $(J_values[end])")
    end

    for (idx, Je) in enumerate(J_values)
        p_current = with_current(p, Je)
        cache = TDGLThermalCache(p_current)
        vector_to_state!(cache.state, u0, p_current)

        if idx > 1
            tspan = (0.0, dt_rampup)
            prob = ODEProblem(fdm_rhs_thermal!, u0, tspan, cache)
            sol = solve(prob, Tsit5(), abstol=1e-5, reltol=1e-5)
            u0 = sol.u[end]
        end

        fastscan_up[idx] = copy(u0)

        # Callback: state is ready for this current point (up direction)
        on_state_ready(idx, Je, copy(u0), :up)

        if verbose
            du_check = similar(u0)
            fdm_rhs_thermal!(du_check, u0, cache, 0.0)
            V_check = compute_voltage_from_du(du_check, p_current)
            vector_to_state!(cache.state, u0, p_current)
            psi_avg = compute_average_order_parameter(cache.state)
            T_avg = compute_average_temperature(cache.state)
            msg = Printf.format(Printf.Format("  [UP] %2d/%2d Je=%.4f: V=%.4e, <|ψ|>=%.4f, <T>=%.4f\n"),
                               idx, n_points, Je, V_check, psi_avg, T_avg)
            write(stdout, msg)
        end
    end

    # Ramp Down
    if verbose
        println("[FastScan Pipeline] Ramp Down: $(J_values[end]) → 0")
    end

    u0 = copy(fastscan_up[end])

    for (idx, Je) in enumerate(reverse(J_values))
        store_idx = n_points - idx + 1
        p_current = with_current(p, Je)
        cache = TDGLThermalCache(p_current)
        vector_to_state!(cache.state, u0, p_current)

        if idx > 1
            tspan = (0.0, dt_rampdown)
            prob = ODEProblem(fdm_rhs_thermal!, u0, tspan, cache)
            sol = solve(prob, Tsit5(), abstol=1e-5, reltol=1e-5)
            u0 = sol.u[end]
        end

        fastscan_down[store_idx] = copy(u0)

        # Callback: state is ready for this current point (down direction)
        on_state_ready(store_idx, Je, copy(u0), :down)

        if verbose
            du_check = similar(u0)
            fdm_rhs_thermal!(du_check, u0, cache, 0.0)
            V_check = compute_voltage_from_du(du_check, p_current)
            vector_to_state!(cache.state, u0, p_current)
            psi_avg = compute_average_order_parameter(cache.state)
            T_avg = compute_average_temperature(cache.state)
            msg = Printf.format(Printf.Format("  [DN] %2d/%2d Je=%.4f: V=%.4e, <|ψ|>=%.4f, <T>=%.4f\n"),
                               idx, n_points, Je, V_check, psi_avg, T_avg)
            write(stdout, msg)
        end
    end

    return fastscan_up, fastscan_down
end

# ============================================================================
# Precompilation Function
# ============================================================================
function precompile_workers(p::TDGLThermalParams, Je_test::Float64)
    println("  Worker $(myid()) precompiling...")

    # Create small test problem
    p_test = with_current(p, Je_test)
    cache = TDGLThermalCache(p_test)
    state = initialize_state_clean(p_test)
    u0 = state_to_vector(state)

    # Solve for short time to trigger compilation
    tspan = (0.0, 1.0)
    prob = ODEProblem(fdm_rhs_thermal!, u0, tspan, cache)
    sol = solve(prob, Tsit5(), abstol=1e-4, reltol=1e-4, maxiters=100)

    # Test H5 writing
    h5open(tempname(), "w") do file
        file["test"] = 1.0
    end

    println("  Worker $(myid()) ready!")
    return true
end

# ============================================================================
# Distributed Worker Function (must be in @everywhere for @printf access)
# ============================================================================
function run_current_point_distributed(p_base, Je::Float64,
                                                    u0_fastscan::Vector,
                                                    is_up::Bool,
                                                    stable_time::Float64,
                                                    dt_snapshots::Float64,
                                                    output_dir::String,
                                                    abstol::Float64,
                                                    reltol::Float64)
    direction = is_up ? "up" : "dn"
    worker_id = myid()

    # Reconstruct params with current
    p = with_current(p_base, Je)
    cache = TDGLThermalCache(p)

    # Initialize state from fastscan
    u0 = copy(u0_fastscan)
    vector_to_state!(cache.state, u0, p)

    # Run simulation
    tspan = (0.0, stable_time)
    prob = ODEProblem(fdm_rhs_thermal!, u0, tspan, cache)
    sol = solve(prob, Tsit5(), abstol=abstol, reltol=reltol,
                saveat=dt_snapshots, progress=true)

    # Collect data
    n_snap = length(sol.t)
    psi_snapshots = zeros(ComplexF64, p.Nx+1, p.Ny+1, n_snap)
    T_snapshots = zeros(Float64, p.Nx+1, p.Ny+1, n_snap)
    V_values = zeros(Float64, n_snap)
    V2_values = zeros(Float64, n_snap)
    psi_avg_values = zeros(Float64, n_snap)
    T_avg_values = zeros(Float64, n_snap)
    times = collect(sol.t)

    du = similar(sol.u[1])
    for (idx, t) in enumerate(sol.t)
        fdm_rhs_thermal!(du, sol.u[idx], cache, t)
        V_values[idx] = compute_voltage_from_du(du, p)
        V2_values[idx] = compute_voltage_hole_from_du(du, p)
        vector_to_state!(cache.state, sol.u[idx], p)
        apply_boundary_conditions!(cache.state, p)  # Apply BCs before saving snapshots
        psi_snapshots[:, :, idx] = cache.state.psi
        T_snapshots[:, :, idx] = cache.state.T
        psi_avg_values[idx] = compute_average_order_parameter(cache.state)
        T_avg_values[idx] = mean(cache.state.T[2:end-1, 2:end-1])
    end

    # Save to H5
    h5_filename = Printf.format(Printf.Format("current_Je%.4f_%s.h5"), Je, direction)
    h5_path = joinpath(output_dir, h5_filename)
    try
        h5open(h5_path, "w") do file
            file["Je"] = Je
            file["direction"] = direction
            file["stable_time"] = stable_time
            file["dt_snapshots"] = dt_snapshots
            file["times"] = times
            file["V"] = V_values
            file["V2"] = V2_values
            file["psi_avg"] = psi_avg_values
            file["T_avg"] = T_avg_values
            file["psi_real"] = real.(psi_snapshots)
            file["psi_imag"] = imag.(psi_snapshots)
            file["T"] = T_snapshots
        end
        # Use Printf.format instead of @printf to avoid world age issues
        msg = Printf.format(Printf.Format("[Worker %d] Je=%.4f (%s): Saved %s\n"), worker_id, Je, direction, h5_filename)
        write(stdout, msg)
    catch e
        msg = Printf.format(Printf.Format("[Worker %d] Error saving %s: %s\n"), worker_id, h5_filename, e)
        write(stdout, msg)
    end

    return Dict(
        :Je => Je,
        :is_up => is_up,
        :h5_file => h5_filename
    )
end

# ============================================================================
# Stable Simulation Function
# ============================================================================
"""
    run_stable_simulation(state0, p, stable_time, dt_snapshots)

Run a stable simulation and collect snapshots.

# Arguments
- `state0::TDGLThermalState`: Initial state
- `p::TDGLThermalParams`: Simulation parameters
- `stable_time::Float64`: Total simulation time
- `dt_snapshots::Float64`: Time interval between snapshots

# Returns
- `psi_res::Matrix{ComplexF64}`: Order parameter snapshots (reshaped to 3D)
- `Ax_res::Matrix{Float64}`: Vector potential snapshots (reshaped to 3D)
- `Ay_res::Matrix{Float64}`: Vector potential snapshots (reshaped to 3D)
- `T_res::Matrix{Float64}`: Temperature snapshots (reshaped to 3D)
- `times::Vector{Float64}`: Time points for snapshots
"""
function run_stable_simulation(state0::TDGLThermalState, p::TDGLThermalParams,
                            stable_time::Float64, dt_snapshots::Float64)
    # Create cache for this simulation
    cache = TDGLThermalCache(p)

    # Convert initial state to vector
    u0 = state_to_vector(state0)

    # Solve ODE
    tspan = (0.0, stable_time)
    prob = ODEProblem(fdm_rhs_thermal!, u0, tspan, cache)
    sol = solve(prob, Tsit5(), abstol=1e-4, reltol=1e-4,
                saveat=dt_snapshots, progress=false)

    # Extract snapshots from solution
    n_snap = length(sol.t)
    nx, ny = p.Nx, p.Ny
    len_psi = (nx+1) * (ny+1)
    len_Ax = nx * (ny+1)
    len_Ay = (nx+1) * ny
    len_psi_plus_Ax = len_psi + len_Ax

    # Reshape to 3D arrays
    # sol.u is a Vector{Vector}, each element is a snapshot
    psi_res = zeros(ComplexF64, nx+1, ny+1, n_snap)
    Ax_res = zeros(Float64, nx, ny+1, n_snap)
    Ay_res = zeros(Float64, nx+1, ny, n_snap)
    T_res = zeros(Float64, nx+1, ny+1, n_snap)

    for i in 1:n_snap
        u_i = sol.u[i]
        psi_i = reshape(u_i[1:len_psi], (nx+1, ny+1))
        Ax_i = reshape(u_i[len_psi+1:len_psi+len_Ax], (nx, ny+1))
        Ay_i = reshape(u_i[len_psi_plus_Ax+1:len_psi_plus_Ax+len_Ay], (nx+1, ny))
        T_i = reshape(u_i[end-len_psi+1:end], (nx+1, ny+1))

        psi_res[:,:,i] = psi_i
        Ax_res[:,:,i] = Ax_i
        Ay_res[:,:,i] = Ay_i
        T_res[:,:,i] = T_i
    end

    times = collect(sol.t)

    return psi_res, Ax_res, Ay_res, T_res, times
end

# ============================================================================
# Simulate Point Function (Worker)
# ============================================================================
"""
    simulate_point(J, p_base, config, direction, output_dir)

Simulate a single current point and compute voltage.

# Arguments
- `J::Float64`: Current density value
- `p_base::TDGLThermalParams`: Base simulation parameters
- `config::SweepConfig`: Sweep configuration
- `direction::String`: Sweep direction ("up" or "down")
- `output_dir::String`: Directory where the per-point `.h5` file will be written

# Returns
- `Dict`: Dictionary with :J, :psi, :T, :V, :times, :h5_file
"""
function save_point_h5(output_dir::String, Je::Float64, direction::String,
                       times, V_values, psi_avg_values, T_avg_values, psi_res, T_res,
                       Ax_res, Ay_res,
                       skip_ratio::Float64)
    h5_filename = Printf.format(Printf.Format("current_Je%.4f_%s.h5"), Je, direction)
    h5_path = joinpath(output_dir, h5_filename)
    stored = trim_snapshot_data_for_storage(times, V_values, psi_res, T_res, skip_ratio, Ax_res, Ay_res)
    stored_psi_avg = psi_avg_values[stored.skip_idx:end]
    stored_T_avg = T_avg_values[stored.skip_idx:end]
    stored_Ax, stored_Ay = stored.extras

    h5open(h5_path, "w") do file
        file["Je"] = Je
        file["direction"] = direction
        file["times"] = stored.times
        file["V"] = stored.V
        file["psi_avg"] = stored_psi_avg
        file["T_avg"] = stored_T_avg
        file["psi_real"] = real.(stored.psi)
        file["psi_imag"] = imag.(stored.psi)
        file["T"] = stored.T
        file["Ax"] = stored_Ax
        file["Ay"] = stored_Ay
        file["stored_skip_idx"] = stored.skip_idx
        file["original_n_snap"] = stored.original_n_snap
    end

    return h5_filename
end

function simulate_point(J::Float64, p_base::TDGLThermalParams, config::SweepConfig, direction::String,
                        output_dir::String; initial_state_override=nothing)
    # Select initial conditions based on direction
    if direction == "up"
        psi_init = config.psi_up_init
        T_init = config.T_initial_up
    else
        psi_init = config.psi_down_init
        T_init = config.T_initial_down
    end

    # Create params for this current value
    p = with_current(p_base, J)

    # Initialize state with specified initial conditions
    state = if isnothing(initial_state_override)
        initialize_state_with_params(p, psi_init, T_init)
    else
        initialize_state_from_fields(p, initial_state_override.psi, initial_state_override.Ax,
                                     initial_state_override.Ay, initial_state_override.T)
    end

    # Run stable simulation
    psi_res, Ax_res, Ay_res, T_res, times = run_stable_simulation(
        state, p, config.stable_time, config.dt_snapshots
    )

    # Calculate voltage from time derivative of Ax
    # Electric field E = -∂A/∂t, Voltage V = E × L
    # Use the last few snapshots for averaging (skip_ratio)
    n_snapshots = length(times)
    skip_idx = max(1, ceil(Int, n_snapshots * config.skip_ratio))

    # Get time step from config
    dt = config.dt_snapshots

    # Compute voltage and averages for all snapshots, then average the tail.
    V_trace = zeros(Float64, n_snapshots)
    psi_avg_values = zeros(Float64, n_snapshots)
    T_avg_values = zeros(Float64, n_snapshots)
    for i in 1:n_snapshots
        psi_avg_values[i] = mean(abs2.(psi_res[2:end-1, 2:end-1, i]))
        T_avg_values[i] = mean(T_res[2:end-1, 2:end-1, i])

        # Calculate voltage from time derivative of Ax
        if i == 1
            # First time step: use forward difference
            Ax_current = Ax_res[:, :, i]
            Ax_next = Ax_res[:, :, i+1]
            E_x = -(Ax_next[2:end-1, 2:end-1] - Ax_current[2:end-1, 2:end-1]) / dt
        elseif i < n_snapshots
            # Middle time steps: use central difference
            Ax_prev = Ax_res[:, :, i-1]
            Ax_current = Ax_res[:, :, i]
            Ax_next = Ax_res[:, :, i+1]
            E_x = -(Ax_next[2:end-1, 2:end-1] - Ax_prev[2:end-1, 2:end-1]) / (2 * dt)
        else
            # Last time step: use backward difference
            Ax_prev = Ax_res[:, :, i-1]
            Ax_current = Ax_res[:, :, i]
            E_x = -(Ax_current[2:end-1, 2:end-1] - Ax_prev[2:end-1, 2:end-1]) / dt
        end

        # Voltage = average electric field × length
        V_trace[i] = mean(E_x) * (p.Nx * p.hx)
    end
    V_avg = n_snapshots >= skip_idx ? mean(V_trace[skip_idx:end]) : 0.0
    h5_file = save_point_h5(output_dir, J, direction, times, V_trace,
                            psi_avg_values, T_avg_values, psi_res, T_res,
                            Ax_res, Ay_res, config.skip_ratio)

    return Dict(
        :J => J,
        :psi => psi_res,
        :T => T_res,
        :V => V_avg,
        :times => times,
        :h5_file => h5_file
    )
end

# Load continuation state from H5 file - must be in @everywhere for workers
function load_continuation_state(h5_path::String)
    isfile(h5_path) || error("Continuation H5 file not found: $h5_path")

    h5open(h5_path, "r") do file
        required = ("psi_real", "psi_imag", "T", "Ax", "Ay")
        for key in required
            haskey(file, key) || error("Continuation H5 file is missing dataset '$key': $h5_path")
        end

        psi = read(file, "psi_real") .+ im .* read(file, "psi_imag")
        T = read(file, "T")
        Ax = read(file, "Ax")
        Ay = read(file, "Ay")

        return (
            psi = psi[:, :, end],
            T = T[:, :, end],
            Ax = Ax[:, :, end],
            Ay = Ay[:, :, end],
        )
    end
end

end # @everywhere - All worker functions defined

# ============================================================================
# Load on main process only (for plotting)
# ============================================================================
using Dates
using Plots
using YAML

gr()

# ============================================================================
# Worker Warmup
# ============================================================================
function warmup_workers(p::TDGLThermalParams)
    println("\n" * "=" ^ 60)
    println("Precompiling workers...")
    println("=" ^ 60)

    Je_test = 0.1  # Small current for fast precompilation
    futures = [@spawnat w precompile_workers(p, Je_test) for w in workers()]

    for f in futures
        fetch(f)
    end

    println("All workers precompiled!")
    println("=" ^ 60)
end

# ============================================================================
# Load Config from YAML (optional)
# ============================================================================
function load_config_yaml(path::String)
    if !isfile(path)
        return nothing
    end
    return YAML.load_file(path)
end

function get_param(cfg, section::String, key::String, default)
    if cfg === nothing
        return default
    end
    sec = get(cfg, section, Dict())
    return get(sec, key, default)
end

# ============================================================================
# Save Config to YAML
# ============================================================================
function save_config_yaml(path::String, p, config)
    open(path, "w") do f
        println(f, "# TDGL + Thermal No-FastScan Configuration")
        println(f, "# Auto-generated from run parameters")
        println(f, "")
        println(f, "grid:")
        println(f, "  Nx: $(p.Nx)")
        println(f, "  Ny: $(p.Ny)")
        println(f, "  hx: $(p.hx)")
        println(f, "  hy: $(p.hy)")
        println(f, "")
        println(f, "tdgl:")
        println(f, "  kappa: $(p.kappa)")
        println(f, "  # sigma1/2/3: conductivity for three regions (uses same gaps as Tc)")
        println(f, "  sigma1: $(p.sigma1)  # bottom region")
        println(f, "  sigma2: $(p.sigma2)  # middle region")
        println(f, "  sigma3: $(p.sigma3)  # top region")
        println(f, "  u: $(p.u)")
        println(f, "  gamma: $(p.gamma)")
        println(f, "  # Three Tc regions: bottom (Tc1), middle (Tc2), top (Tc3)")
        println(f, "  Tc1: $(p.Tc1)")
        println(f, "  Tc2: $(p.Tc2)")
        println(f, "  Tc3: $(p.Tc3)")
        println(f, "  Tc1gap: $(p.Tc1gap)  # fraction of height for Tc1 at bottom")
        println(f, "  Tc3gap: $(p.Tc3gap)  # fraction of height for Tc3 at top")
        println(f, "")
        println(f, "thermal:")
        println(f, "  # enabled: if false, disable thermal coupling (T stays at T0)")
        println(f, "  enabled: $(p.thermal_enabled)")
        println(f, "  C_eff: $(p.C_eff)")
        println(f, "  k_eff: $(p.k_eff)")
        println(f, "  T0: $(p.T0)")
        println(f, "  # epsilon_type: GL coupling ε(T)")
        println(f, "  #   linear - ε = Tc - T (default)")
        println(f, "  #   bcs    - ε = Tc/T - 1")
        println(f, "  epsilon_type: $(p.epsilon_type)")
        println(f, "  eta_env: $(p.eta_env)")
        println(f, "  eta_hole: $(p.eta_hole)")
        println(f, "  holewidth: $(p.holewidth)")
        println(f, "")
        println(f, "fields:")
        println(f, "  He: $(p.He)")
        println(f, "")
        println(f, "boundary:")
        println(f, "  # bc_x options: periodic, neumann, dirichlet (psi=1, T=T0 at left/right)")
        println(f, "  bc_x: $(p.bc_x)")
        println(f, "  # bc_x_induced_field_scale: current-induced magnetic field on left/right boundaries")
        println(f, "  #   actual x-boundary field = bc_x_induced_field_scale * Je")
        println(f, "  bc_x_induced_field_scale: $(p.bc_x_induced_field_scale)")
        println(f, "  # bc_y options: neumann (covariant Neumann for psi), dirichlet (psi=0 at electrode)")
        println(f, "  # Temperature always adiabatic at y-boundaries (unless loop=true)")
        println(f, "  bc_y: $(p.bc_y)")
        println(f, "  # bc_y_coverage: electrode coverage fraction (0-1), centered")
        println(f, "  #   1.0 = full boundary covered by electrode")
        println(f, "  #   0.5 = middle half covered, left/right edges exposed")
        println(f, "  # Magnetic field (H_induced from current) only applies in electrode region")
        println(f, "  bc_y_coverage: $(p.bc_y_coverage)")
        println(f, "  # loop: if true, non-electrode y-boundaries form periodic loops (top-bottom)")
        println(f, "  #   Left side loops with left side, right side loops with right side")
        println(f, "  loop: $(p.loop)")
        println(f, "  # field_side: which boundary gets current-induced magnetic field")
        println(f, "  #   both  - symmetric, both y-boundaries get ±H_induced (default)")
        println(f, "  #   upper - only upper boundary gets field, lower is zero")
        println(f, "  #   lower - only lower boundary gets field, upper is zero")
        println(f, "  field_side: $(p.field_side)")
        println(f, "  # psi_y_zero: if true, psi=0 at all y boundaries (NS contact)")
        println(f, "  psi_y_zero: $(p.psi_y_zero)")
        println(f, "")
        println(f, "noise:")
        println(f, "  # Langevin noise for Cooper pair nucleation")
        println(f, "  # Helps recovery from normal to superconducting state")
        println(f, "  strength: $(p.noise_strength)")
        println(f, "")
        println(f, "sweep:")
        println(f, "  Jpeak: $(config.Jpeak)")
        println(f, "  n_current_points: $(config.n_current_points)")
        # Note: rampup_time and rampdown_time are kept for compatibility but not used in nofastscan mode
        println(f, "  # rampup_time: $(config.rampup_time)  # Not used in nofastscan mode")
        println(f, "  # rampdown_time: $(config.rampdown_time)  # Not used in nofastscan mode")
        println(f, "  stable_time: $(config.stable_time)")
        println(f, "  dt_snapshots: $(config.dt_snapshots)")
        println(f, "  skip_ratio: $(config.skip_ratio)")
        println(f, "  run_up: $(config.run_up)")
        println(f, "  run_down: $(config.run_down)")
        println(f, "  input_folder: $(isnothing(config.input_folder) ? "none" : config.input_folder)")
        println(f, "  # Initial conditions for nofastscan mode")
        println(f, "  psi_up_init: $(config.psi_up_init)")
        println(f, "  T_initial_up: $(config.T_initial_up)")
        println(f, "  psi_down_init: $(config.psi_down_init)")
        println(f, "  T_initial_down: $(config.T_initial_down)")
        println(f, "")
        println(f, "outputs:")
        println(f, "  generate_mp4: $(config.generate_mp4)")
        println(f, "  mp4_fps: $(config.mp4_fps)")
        println(f, "  frames_per_point: $(config.frames_per_point)")
    end
end

function save_refine_yaml(path::String, config)
    open(path, "w") do f
        println(f, "# Refine Configuration")
        println(f, "# Edit these parameters and run: julia -p N current_sweep_thermal_nofastscan.jl <this_folder>")
        println(f, "")
        println(f, "refine:")
        println(f, "  Jstart: $(config.Jpeak * 0.3)      # Start of refine range")
        println(f, "  Jend: $(config.Jpeak * 0.5)        # End of refine range")
        println(f, "  refine_num: 3                      # Points to insert BETWEEN each pair of adjacent original points")
        println(f, "  stable_time: $(config.stable_time)   # Same as original to skip recalc")
        println(f, "  dt_snapshots: $(config.dt_snapshots)  # Same as original to skip recalc")
        println(f, "  skip_ratio: $(config.skip_ratio)      # Skip first X% when computing averages")
        println(f, "  run_up: $(config.run_up)")
        println(f, "  run_down: $(config.run_down)")
        println(f, "  # Initial conditions")
        println(f, "  psi_up_init: $(config.psi_up_init)")
        println(f, "  T_initial_up: $(config.T_initial_up)")
        println(f, "  psi_down_init: $(config.psi_down_init)")
        println(f, "  T_initial_down: $(config.T_initial_down)")
    end
end

# ============================================================================
# Build Parameters
# ============================================================================
function build_params_from_config(cfg)
    # Get sigma values - use nothing if not specified to trigger uniform sigma
    # Handle case where cfg is Nothing or invalid
    if cfg === nothing
        return build_params_from_defaults()
    end

    sigma_base = get_param(cfg, "tdgl", "sigma", PARAM_sigma)
    sigma1_cfg = haskey(cfg, "tdgl") && haskey(cfg["tdgl"], "sigma1") ? cfg["tdgl"]["sigma1"] : nothing
    sigma2_cfg = haskey(cfg, "tdgl") && haskey(cfg["tdgl"], "sigma2") ? cfg["tdgl"]["sigma2"] : nothing
    sigma3_cfg = haskey(cfg, "tdgl") && haskey(cfg["tdgl"], "sigma3") ? cfg["tdgl"]["sigma3"] : nothing

    return TDGLThermalParams(
        Nx = get_param(cfg, "grid", "Nx", PARAM_Nx),
        Ny = get_param(cfg, "grid", "Ny", PARAM_Ny),
        hx = get_param(cfg, "grid", "hx", PARAM_hx),
        hy = get_param(cfg, "grid", "hy", PARAM_hy),
        kappa = get_param(cfg, "tdgl", "kappa", PARAM_kappa),
        sigma = sigma_base,
        sigma1 = sigma1_cfg,
        sigma2 = sigma2_cfg,
        sigma3 = sigma3_cfg,
        u = get_param(cfg, "tdgl", "u", PARAM_u),
        gamma = get_param(cfg, "tdgl", "gamma", PARAM_gamma),
        C_eff = get_param(cfg, "thermal", "C_eff", PARAM_C_eff),
        k_eff = get_param(cfg, "thermal", "k_eff", PARAM_k_eff),
        T0 = get_param(cfg, "thermal", "T0", PARAM_T0),
        eta_env = get_param(cfg, "thermal", "eta_env", PARAM_eta_env),
        eta_hole = get_param(cfg, "thermal", "eta_hole", PARAM_eta_hole),
        holewidth = get_param(cfg, "thermal", "holewidth", PARAM_holewidth),
        Tc1 = get_param(cfg, "tdgl", "Tc1", PARAM_Tc1),
        Tc2 = get_param(cfg, "tdgl", "Tc2", PARAM_Tc2),
        Tc3 = get_param(cfg, "tdgl", "Tc3", PARAM_Tc3),
        Tc1gap = get_param(cfg, "tdgl", "Tc1gap", PARAM_Tc1gap),
        Tc3gap = get_param(cfg, "tdgl", "Tc3gap", PARAM_Tc3gap),
        He = get_param(cfg, "fields", "He", PARAM_He),
        bc_x = Symbol(get_param(cfg, "boundary", "bc_x", string(PARAM_bc_x))),
        bc_x_induced_field_scale = get_param(cfg, "boundary", "bc_x_induced_field_scale", PARAM_bc_x_induced_field_scale),
        bc_y = Symbol(get_param(cfg, "boundary", "bc_y", string(PARAM_bc_y))),
        bc_y_coverage = get_param(cfg, "boundary", "bc_y_coverage", PARAM_bc_y_coverage),
        loop = get_param(cfg, "boundary", "loop", PARAM_loop),
        field_side = Symbol(get_param(cfg, "boundary", "field_side", string(PARAM_field_side))),
        psi_y_zero = get_param(cfg, "boundary", "psi_y_zero", PARAM_psi_y_zero),
        thermal_enabled = get_param(cfg, "thermal", "enabled", PARAM_thermal_enabled),
        epsilon_type = Symbol(get_param(cfg, "thermal", "epsilon_type", string(PARAM_epsilon_type))),
        noise_strength = get_param(cfg, "noise", "strength", PARAM_noise_strength)
    )
end

function build_params_from_defaults()
    # Build parameters from default values
    sigma_base = PARAM_sigma
    return TDGLThermalParams(
        Nx = PARAM_Nx,
        Ny = PARAM_Ny,
        hx = PARAM_hx,
        hy = PARAM_hy,
        kappa = PARAM_kappa,
        sigma = sigma_base,
        sigma1 = sigma_base,
        sigma2 = sigma_base,
        sigma3 = sigma_base,
        u = PARAM_u,
        gamma = PARAM_gamma,
        C_eff = PARAM_C_eff,
        k_eff = PARAM_k_eff,
        T0 = PARAM_T0,
        eta_env = PARAM_eta_env,
        eta_hole = PARAM_eta_hole,
        holewidth = PARAM_holewidth,
        Tc1 = PARAM_Tc1,
        Tc2 = PARAM_Tc2,
        Tc3 = PARAM_Tc3,
        Tc1gap = PARAM_Tc1gap,
        Tc3gap = PARAM_Tc3gap,
        He = PARAM_He,
        bc_x = Symbol(string(PARAM_bc_x)),
        bc_x_induced_field_scale = PARAM_bc_x_induced_field_scale,
        bc_y = Symbol(string(PARAM_bc_y)),
        bc_y_coverage = PARAM_bc_y_coverage,
        loop = PARAM_loop,
        field_side = Symbol(string(PARAM_field_side)),
        psi_y_zero = PARAM_psi_y_zero,
        thermal_enabled = PARAM_thermal_enabled,
        epsilon_type = Symbol(string(PARAM_epsilon_type)),
        noise_strength = PARAM_noise_strength
    )
end

function build_sweep_config(cfg; input_folder_from_cmd=nothing, skip_ratio_override=nothing, stable_time_override=nothing)
    Jpeak = get_param(cfg, "sweep", "Jpeak", PARAM_Jpeak)
    n_current_points = get_param(cfg, "sweep", "n_current_points", PARAM_n_current_points)
    rampup_time = get_param(cfg, "sweep", "rampup_time", PARAM_rampup_time)
    rampdown_time = get_param(cfg, "sweep", "rampdown_time", PARAM_rampdown_time)
    stable_time = get_param(cfg, "sweep", "stable_time", PARAM_stable_time)
    dt_snapshots = get_param(cfg, "sweep", "dt_snapshots", PARAM_dt_snapshots)
    skip_ratio = get_param(cfg, "sweep", "skip_ratio", PARAM_skip_ratio)
    run_up = get_param(cfg, "sweep", "run_up", true)
    run_down = get_param(cfg, "sweep", "run_down", true)
    generate_mp4 = get_param(cfg, "outputs", "generate_mp4", false)
    mp4_fps = Int(get_param(cfg, "outputs", "mp4_fps", 5))
    frames_per_point = Int(get_param(cfg, "outputs", "frames_per_point", 20))

    # NEW: Initial conditions with defaults
    psi_up_init = get_param(cfg, "sweep", "psi_up_init", 1.0)
    T_initial_up = get_param(cfg, "sweep", "T_initial_up", 0.0)
    psi_down_init = get_param(cfg, "sweep", "psi_down_init", 0.0)
    T_initial_down = get_param(cfg, "sweep", "T_initial_down", 1.2)

    # input_folder from command line takes priority over config
    input_folder = if !isnothing(input_folder_from_cmd)
        value = strip(String(input_folder_from_cmd))
        lowercase(value) in ("", "none", "nothing", "null") ? nothing : value
    else
        raw_input_folder = get_param(cfg, "sweep", "input_folder", nothing)
        if isnothing(raw_input_folder)
            nothing
        else
            value = strip(String(raw_input_folder))
            lowercase(value) in ("", "none", "nothing", "null") ? nothing : value
        end
    end

    # Apply stable_time override if provided, or use default 1000 for continuation mode
    if !isnothing(stable_time_override)
        stable_time = stable_time_override
        println("Using overridden stable_time: $stable_time")
    elseif !isnothing(input_folder_from_cmd)
        # Default stable_time=1000 for continuation runs (only when --from flag was used)
        stable_time = 1000.0
        println("Continuation mode: using default stable_time=1000")
    end

    # Apply skip_ratio override if provided, or use default 0.5 for continuation mode
    if !isnothing(skip_ratio_override)
        skip_ratio = skip_ratio_override
        println("Using overridden skip_ratio: $skip_ratio")
    elseif !isnothing(input_folder)
        # Default skip_ratio=0.5 for continuation runs (starting from stable state)
        skip_ratio = 0.5
        println("Continuation mode: using default skip_ratio=0.5")
    end

    if isnothing(input_folder) && !run_up && !run_down
        error("At least one sweep direction must be enabled: set sweep.run_up or sweep.run_down to true")
    end

    return SweepConfig(Jpeak, n_current_points, rampup_time, rampdown_time,
                     stable_time, dt_snapshots, skip_ratio, run_up, run_down,
                     generate_mp4, mp4_fps, frames_per_point,
                     psi_up_init, T_initial_up, psi_down_init, T_initial_down,
                     input_folder)
end

function discover_input_folder_points(input_folder::String)
    isdir(input_folder) || error("input_folder does not exist or is not a directory: $input_folder")

    # Check if H5 files exist directly in the input folder
    parsed = NamedTuple[]
    for name in readdir(input_folder)
        m = match(r"^current_Je([0-9.]+)_(up|down)\.h5$", name)
        m === nothing && continue

        push!(parsed, (
            Je = parse(Float64, m.captures[1]),
            direction = m.captures[2],
            file = joinpath(input_folder, name),
        ))
    end

    # If no H5 files found directly, look for nested output folder
    if isempty(parsed)
        println("No H5 files found directly in $input_folder")
        println("Looking for nested output folder...")
        for name in readdir(input_folder)
            m = match(r"^sweep_nofastscan_", name)
            if m !== nothing && isdir(joinpath(input_folder, name))
                nested_folder = joinpath(input_folder, name)
                println("Found nested output folder: $nested_folder")
                # Recursively call with nested folder
                return discover_input_folder_points(nested_folder)
            end
        end
        error("No continuation H5 files found in $input_folder or its nested folders")
    end

    up_points = sort(filter(point -> point.direction == "up", parsed); by = point -> point.Je)
    down_points = sort(filter(point -> point.direction == "down", parsed); by = point -> point.Je, rev = true)

    return up_points, down_points
end

# ============================================================================
# Pipeline Mode: FastScan + Distributed Stable run concurrently
# ============================================================================
function run_pipeline_mode(p, config, output_dir::String, J_values)
    println("\n" * "=" ^ 60)
    println("Pipeline Mode: FastScan + Stable concurrent execution")
    println("  Rampup time: $(config.rampup_time)")
    println("  Rampdown time: $(config.rampdown_time)")
    println("  Stable time: $(config.stable_time)")
    println("  dt_snapshots: $(config.dt_snapshots)")
    println("  Workers: $(nworkers())")
    println("=" ^ 60)

    abs_output_dir = abspath(output_dir)
    n_points = length(J_values)

    # Storage for futures (async tasks)
    futures_up = Vector{Any}(undef, n_points)
    futures_down = Vector{Any}(undef, n_points)

    # Track dispatched tasks
    dispatched_up = 0
    dispatched_down = 0

    # Callback function: called when each fastscan state is ready
    function on_state_ready(idx::Int, Je::Float64, u0::Vector, phase::Symbol)
        if phase == :up
            # Dispatch stable run for UP direction immediately
            futures_up[idx] = @spawnat :any run_current_point_distributed(
                p, Je, u0, true,
                config.stable_time, config.dt_snapshots,
                abs_output_dir, PARAM_abstol, PARAM_reltol
            )
            dispatched_up += 1
            @printf("  [DISPATCH UP] %2d/%2d Je=%.4f → worker\n", dispatched_up, n_points, Je)
        else
            # Dispatch stable run for DOWN direction immediately
            futures_down[idx] = @spawnat :any run_current_point_distributed(
                p, Je, u0, false,
                config.stable_time, config.dt_snapshots,
                abs_output_dir, PARAM_abstol, PARAM_reltol
            )
            dispatched_down += 1
            @printf("  [DISPATCH DN] %2d/%2d Je=%.4f → worker\n", dispatched_down, n_points, Je)
        end
    end

    # Run fastscan with pipeline callback
    println("\n[Pipeline] Starting fastscan with concurrent stable dispatch...")
    fastscan_up, fastscan_down = run_fastscan_pipeline(
        p, J_values,
        config.rampup_time, config.rampdown_time,
        on_state_ready;
        verbose=true
    )

    # Save fastscan states (for reference/debugging)
    h5open(joinpath(output_dir, "fastscan_states.h5"), "w") do file
        file["J_values"] = J_values
        file["n_points"] = length(J_values)
        for (idx, Je) in enumerate(J_values)
            file["up_$(idx)_real"] = real.(fastscan_up[idx])
            file["up_$(idx)_imag"] = imag.(fastscan_up[idx])
            file["down_$(idx)_real"] = real.(fastscan_down[idx])
            file["down_$(idx)_imag"] = imag.(fastscan_down[idx])
        end
    end
    println("\nSaved: fastscan_states.h5")

    # Wait for all stable runs to complete
    println("\n[Pipeline] Waiting for all stable runs to complete...")
    println("  Total tasks: $(2 * n_points) ($(n_points) up + $(n_points) down)")

    results_up = Vector{Dict}(undef, n_points)
    results_down = Vector{Dict}(undef, n_points)

    # Collect results with progress
    completed = 0
    total = 2 * n_points

    for idx in 1:n_points
        results_up[idx] = fetch(futures_up[idx])
        completed += 1
        @printf("  [COMPLETE] %3d/%3d (up   Je=%.4f)\n", completed, total, J_values[idx])
    end

    for idx in 1:n_points
        results_down[idx] = fetch(futures_down[idx])
        completed += 1
        @printf("  [COMPLETE] %3d/%3d (down Je=%.4f)\n", completed, total, J_values[idx])
    end

    # Sort by Je
    results_up = sort(results_up, by=r->r[:Je])
    results_down = sort(results_down, by=r->r[:Je])

    println("\n[Pipeline] All tasks completed!")

    return results_up, results_down, fastscan_up, fastscan_down
end

# ============================================================================
# Parallel 4-Window Animation Functions
# ============================================================================

# Load data for a single current point from H5 file
function load_single_Je_data(output_dir::String, Je::Float64, direction::String, skip_ratio::Float64)
    h5_file = Printf.format(Printf.Format("current_Je%.4f_%s.h5"), Je, direction)
    h5_path = joinpath(output_dir, h5_file)

    if !isfile(h5_path)
        println("  Warning: $h5_file not found, skipping...")
        return nothing
    end

    h5open(h5_path, "r") do file
        psi_real = read(file, "psi_real")
        psi_imag = read(file, "psi_imag")
        psi = psi_real .+ im .* psi_imag

        T = read(file, "T")
        times = read(file, "times")
        V = read(file, "V")

        aligned = normalize_animation_snapshot_data(times, V, psi, T, skip_ratio; already_trimmed=true)

        return (
            Je = Je,
            direction = direction,
            psi = aligned.psi,
            T = aligned.T,
            times = aligned.times,
            skip_idx = aligned.skip_idx,
            T_avg = aligned.T_avg,
            V = aligned.V,
            V_avg = aligned.V_avg,
            V_t = aligned.V_t
        )
    end
end

@everywhere using Plots

@everywhere function generate_single_Je_animation_remote(frames_dir::String, Je::Float64, data,
                                                         V_t_up::Dict{Float64, Vector{Tuple{Float64,Float64}}},
                                                         V_t_down::Dict{Float64, Vector{Tuple{Float64,Float64}}},
                                                         direction::String, x, y,
                                                         T_max_all::Float64, V_max_all::Float64, Je_max::Float64,
                                                         idx::Int, Je_up, V_up,
                                                         Je_down, V_down,
                                                         Nx::Int, Ny::Int, fps::Int, frames_per_point::Int)
    psi = data.psi
    T = data.T
    times = data.times
    skip_idx = data.skip_idx
    T_avg = data.T_avg
    n_snap = length(times)
    post_skip_indices = collect(skip_idx:n_snap)
    n_render = min(frames_per_point, length(post_skip_indices))
    sample_positions = round.(Int, range(1, length(post_skip_indices), length=n_render))
    sampled_indices = unique(post_skip_indices[sample_positions])

    anim = Animation()

    # Sampled indices affect only rendered spatial frames; voltage statistics still use all post-skip samples.
    for t_idx in sampled_indices
        psi_int = psi[2:Nx, 2:Ny, t_idx]
        T_int = T[2:Nx, 2:Ny, t_idx]
        t_val = times[t_idx]
        T_avg_val = T_avg[t_idx]

        p1 = heatmap(x, y, abs2.(psi_int)',
                     xlabel="x", ylabel="y",
                     title=@sprintf("|ψ|² Je=%.4f t=%.1f", Je, t_val),
                     color=:viridis, clims=(0, 1.1), aspect_ratio=:equal)

        p2 = heatmap(x, y, T_int',
                     xlabel="x", ylabel="y",
                     title=@sprintf("T (%s) T=%.4f", direction, T_avg_val),
                     color=:hot, clims=(0, T_max_all), aspect_ratio=:equal)

        p3 = plot(xlabel="Je", ylabel="V", title="I-V Characteristic",
                  legend=:topleft, xlims=(0, Je_max * 1.1))
        if !isempty(Je_up)
            plot!(p3, collect(Je_up), collect(V_up), label="Up", color=:blue, lw=1, alpha=0.3)
        end
        if !isempty(Je_down)
            plot!(p3, collect(Je_down), collect(V_down), label="Down", color=:red, lw=1, alpha=0.3)
        end
        if direction == "up"
            idx_up = findfirst(j -> j == Je, collect(Je_up))
            if idx_up !== nothing
                scatter!(p3, [Je], [collect(V_up)[idx_up]], color=:blue, ms=6, label="", markerstrokewidth=2)
            end
        else
            idx_down = findfirst(j -> j == Je, collect(Je_down))
            if idx_down !== nothing
                scatter!(p3, [Je], [collect(V_down)[idx_down]], color=:red, ms=6, label="", markerstrokewidth=2)
            end
        end

        p4 = plot(xlabel="t", ylabel="V", title=@sprintf("V vs t (Je=%.3f)", Je),
                  legend=:topright)
        dir_V_t_data = direction == "up" ? V_t_up : V_t_down
        for (Je_val, V_t_array) in sort(collect(dir_V_t_data); by=first)
            if isapprox(Je_val, Je, rtol=1e-6)
                times_array = [t for (t, _) in V_t_array]
                V_array = [v for (_, v) in V_t_array]
                plot!(p4, times_array, V_array, label=@sprintf("Je=%.3f", Je_val), lw=1.5)
            end
        end

        frame(anim, plot(p1, p2, p3, p4, layout=grid(2, 2), size=(1400, 900)))
    end

    mkpath(frames_dir)
    mp4_file = joinpath(frames_dir, Printf.format(Printf.Format("evolution_Je%d.mp4"), idx))
    mp4(anim, mp4_file, fps=fps)
    return mp4_file
end

function generate_parallel_4window_animation(output_dir::String, data_up, data_down,
                                             x, y, T_max_all::Float64, V_max_all::Float64, Je_max::Float64,
                                             Nx::Int, Ny::Int, fps::Int, frames_per_point::Int)
    println("\n" * "=" ^ 60)
    println("Step 4: Generating 4-window MP4 segments")
    println("  fps = $fps")
    println("  Workers = $(nworkers())")
    println("=" ^ 60)

    frames_dir = joinpath(output_dir, "frames")
    mkpath(frames_dir)

    V_t_up = Dict{Float64, Vector{Tuple{Float64, Float64}}}(d.Je => d.V_t for d in data_up)
    V_t_down = Dict{Float64, Vector{Tuple{Float64, Float64}}}(d.Je => d.V_t for d in data_down)

    Je_up = [d.Je for d in data_up]
    V_up = [d.V_avg for d in data_up]
    Je_down = [d.Je for d in data_down]
    V_down = [d.V_avg for d in data_down]

    futures = Any[]
    idx = 1
    for d in data_up
        push!(futures, @spawnat :any Base.invokelatest(
            generate_single_Je_animation_remote,
            frames_dir, d.Je, d, V_t_up, V_t_down, "up",
            x, y, T_max_all, V_max_all, Je_max, idx,
            Je_up, V_up, Je_down, V_down, Nx, Ny, fps, frames_per_point
        ))
        idx += 1
    end
    for d in data_down
        push!(futures, @spawnat :any Base.invokelatest(
            generate_single_Je_animation_remote,
            frames_dir, d.Je, d, V_t_up, V_t_down, "down",
            x, y, T_max_all, V_max_all, Je_max, idx,
            Je_up, V_up, Je_down, V_down, Nx, Ny, fps, frames_per_point
        ))
        idx += 1
    end

    mp4_files = String[]
    for future in futures
        mp4_file = fetch(future)
        if mp4_file !== nothing
            push!(mp4_files, mp4_file)
        end
    end
    return mp4_files
end

function merge_animations_with_ffmpeg(output_dir::String, mp4_files::Vector{String}, fps::Int)
    if isempty(mp4_files)
        println("  No MP4 segments generated; skipping merge")
        return
    end

    frames_dir = joinpath(output_dir, "frames")
    output_path = joinpath(output_dir, "evolution_4window.mp4")
    input_file = joinpath(frames_dir, "input.txt")

    open(input_file, "w") do f
        for mp4_file in mp4_files
            println(f, "file '$(basename(mp4_file))'")
        end
    end

    try
        run(`ffmpeg -f concat -safe 0 -i $input_file -c:v libx264 -pix_fmt yuv420p -r $fps -y $output_path`)
        println("  Saved: evolution_4window.mp4")
    catch e
        println("  FFmpeg error: $e")
        println("  MP4 segments remain in $frames_dir")
    finally
        isfile(input_file) && rm(input_file)
    end
end

# ============================================================================
# Generate Outputs for No-FastScan Mode
# ============================================================================
function generate_outputs_nofastscan(output_dir::String, J_values,
                                    Je_up, V_up, psi_up, T_up,
                                    Je_down, V_down, psi_down, T_down,
                                    p, config)
    println("\n" * "=" ^ 60)
    println("Step 3: Generating Plots and Outputs")
    println("  skip_ratio = $(config.skip_ratio)")
    println("=" ^ 60)

    # Interior coordinates (avoids boundary artifacts)
    Lx = p.Nx * p.hx
    Ly = p.Ny * p.hy
    x_full = range(-Lx/2, Lx/2, length=p.Nx+1)
    y_full = range(-Ly/2, Ly/2, length=p.Ny+1)
    x = x_full[2:p.Nx]
    y = y_full[2:p.Ny]

    # Compute average psi and T from snapshots (skip first part for steady state)
    skip_ratio = config.skip_ratio

    function compute_snapshot_mean(snapshots)
        # snapshots is a vector of 3D arrays
        means = Float64[]
        for snap in snapshots
            n_snap = size(snap, 3)
            skip_idx = max(1, ceil(Int, n_snap * skip_ratio))
            # Compute mean over interior points and time (after skip)
            interior = snap[2:end-1, 2:end-1, skip_idx:end]
            mean_val = mean(abs2.(interior))  # For psi, use |psi|^2
            push!(means, mean_val)
        end
        return means
    end

    # Compute average values for plotting
    # Note: V_up and V_down are already averaged from simulate_point
    # psi_up and T_up are vectors of 3D arrays, need to compute averages
    psi_avg_up = compute_snapshot_mean(psi_up)
    psi_avg_down = compute_snapshot_mean(psi_down)

    # T averages: handle 3D arrays by averaging over interior and time (after skip)
    function compute_T_avg(snapshots)
        means = Float64[]
        for snap in snapshots
            n_snap = size(snap, 3)
            skip_idx = max(1, ceil(Int, n_snap * skip_ratio))
            # Average over interior points and time (after skip)
            interior = snap[2:end-1, 2:end-1, skip_idx:end]
            mean_val = mean(interior)
            push!(means, mean_val)
        end
        return means
    end

    T_avg_up = compute_T_avg(T_up)
    T_avg_down = compute_T_avg(T_down)

    # I-V curve (full domain)
    p_iv = plot(xlabel="Je", ylabel="V", title="I-V Characteristic",
                legend=:topleft, size=(700, 500))
    if !isempty(Je_up)
        plot!(p_iv, Je_up, V_up, label="Up", color=:blue, lw=2, marker=:circle, ms=4)
    end
    if !isempty(Je_down)
        plot!(p_iv, Je_down, V_down, label="Down", color=:red, lw=2, marker=:square, ms=4)
    end
    savefig(p_iv, joinpath(output_dir, "IV_curve.png"))
    println("  Saved: IV_curve.png")

    # Order parameter
    p_psi = plot(xlabel="Je", ylabel="<|ψ|>²", title="Order Parameter",
                 legend=:topright, size=(700, 500))
    if !isempty(Je_up)
        plot!(p_psi, Je_up, psi_avg_up, label="Up", color=:blue, lw=2, marker=:circle, ms=4)
    end
    if !isempty(Je_down)
        plot!(p_psi, Je_down, psi_avg_down, label="Down", color=:red, lw=2, marker=:square, ms=4)
    end
    savefig(p_psi, joinpath(output_dir, "psi_vs_current.png"))
    println("  Saved: psi_vs_current.png")

    # Temperature
    p_T = plot(xlabel="Je", ylabel="<T>", title="Temperature",
               legend=:topleft, size=(700, 500))
    if !isempty(Je_up)
        plot!(p_T, Je_up, T_avg_up, label="Up", color=:blue, lw=2, marker=:circle, ms=4)
    end
    if !isempty(Je_down)
        plot!(p_T, Je_down, T_avg_down, label="Down", color=:red, lw=2, marker=:square, ms=4)
    end
    savefig(p_T, joinpath(output_dir, "T_vs_current.png"))
    println("  Saved: T_vs_current.png")

    # Eta distribution
    eta = compute_eta_distribution(p)
    p_eta = heatmap(x, y, eta[2:p.Nx, 2:p.Ny]',
                    xlabel="x", ylabel="y", title="eta distribution",
                    color=:viridis, aspect_ratio=:equal, size=(550, 500))
    savefig(p_eta, joinpath(output_dir, "eta_distribution.png"))
    println("  Saved: eta_distribution.png")

    # Tc distribution
    Tc = compute_Tc_distribution(p)
    p_Tc = heatmap(x, y, Tc[2:p.Nx, 2:p.Ny]',
                   xlabel="x", ylabel="y", title="Tc distribution",
                   color=:coolwarm, aspect_ratio=:equal, size=(550, 500))
    savefig(p_Tc, joinpath(output_dir, "Tc_distribution.png"))
    println("  Saved: Tc_distribution.png")

    # Parameters.txt
    open(joinpath(output_dir, "parameters.txt"), "w") do f
        println(f, "TDGL + Thermal No-FastScan Sweep")
        println(f, "Date: $(now())")
        println(f, "Workers: $(nworkers())")
        println(f, "Grid: $(p.Nx) x $(p.Ny)")
        println(f, "Tc1=$(p.Tc1), Tc2=$(p.Tc2), Tc3=$(p.Tc3), Tc1gap=$(p.Tc1gap), Tc3gap=$(p.Tc3gap)")
        println(f, "eta_env=$(p.eta_env), eta_hole=$(p.eta_hole), holewidth=$(p.holewidth)")
        println(f, "Jpeak=$(config.Jpeak), n_points=$(config.n_current_points)")
        println(f, "run_up=$(config.run_up), run_down=$(config.run_down)")
        println(f, "input_folder=$(isnothing(config.input_folder) ? "none" : config.input_folder)")
        println(f, "generate_mp4=$(config.generate_mp4), mp4_fps=$(config.mp4_fps)")
        println(f, "Initial Up: ψ=$(config.psi_up_init), T=$(config.T_initial_up)")
        println(f, "Initial Down: ψ=$(config.psi_down_init), T=$(config.T_initial_down)")
        println(f, "stable_time=$(config.stable_time), dt_snapshots=$(config.dt_snapshots), skip_ratio=$(config.skip_ratio)")
    end
    println("  Saved: parameters.txt")

    if config.generate_mp4
        println("\n" * "=" ^ 60)
        println("Step 4: Generating MP4 from per-point H5 files")
        println("  skip_ratio = $(config.skip_ratio)")
        println("=" ^ 60)

        data_up = filter(!isnothing, [load_single_Je_data(output_dir, Je, "up", config.skip_ratio) for Je in Je_up])
        data_down = filter(!isnothing, [load_single_Je_data(output_dir, Je, "down", config.skip_ratio) for Je in Je_down])

        if isempty(data_up) && isempty(data_down)
            println("  No H5 data found for MP4 generation; skipping")
        else
            T_max_all = 0.1
            for d in vcat(data_up, data_down)
                T_max_all = max(T_max_all, maximum(d.T))
            end

            all_V = vcat(V_up, V_down)
            all_Je = vcat(Je_up, Je_down)
            V_max_all = isempty(all_V) ? 0.1 : maximum(all_V)
            Je_max = isempty(all_Je) ? config.Jpeak : maximum(all_Je)

            mp4_files = generate_parallel_4window_animation(output_dir, data_up, data_down,
                                                            x, y, T_max_all, V_max_all, Je_max,
                                                            p.Nx, p.Ny, config.mp4_fps, config.frames_per_point)
            merge_animations_with_ffmpeg(output_dir, mp4_files, config.mp4_fps)
        end
    else
        println("  MP4 generation skipped by config")
    end

    println("\n" * "=" ^ 60)
    println("Results saved to: $(abspath(output_dir))/")
    println("=" ^ 60)
end

# ============================================================================
# Step 3: Generate Plots and GIFs (legacy fastscan - kept for reference)
# ============================================================================
function generate_outputs(output_dir::String, J_values, results_up, results_down, p, config)
    println("\n" * "=" ^ 60)
    println("Step 3: Generating Plots and GIFs")
    println("  skip_ratio = $(config.skip_ratio)")
    println("=" ^ 60)

    # Interior coordinates (avoids boundary artifacts)
    Lx = p.Nx * p.hx
    Ly = p.Ny * p.hy
    x_full = range(-Lx/2, Lx/2, length=p.Nx+1)
    y_full = range(-Ly/2, Ly/2, length=p.Ny+1)
    x = x_full[2:p.Nx]
    y = y_full[2:p.Ny]
    # Load data from H5 files. Per-point H5 data is already post-skip.
    function load_data_with_skip(h5_file)
        h5_path = joinpath(output_dir, h5_file)
        if !isfile(h5_path)
            println("  Warning: File not found: $h5_file")
            return nothing
        end
        h5open(h5_path, "r") do file
            times = read(file, "times")
            V = read(file, "V")
            V2 = haskey(file, "V2") ? read(file, "V2") : zeros(length(V))
            psi_avg = read(file, "psi_avg")
            T_avg = read(file, "T_avg")
            psi_real = read(file, "psi_real")
            psi_imag = read(file, "psi_imag")
            T_snap = read(file, "T")
            Je = read(file, "Je")

            psi = psi_real .+ im .* psi_imag

            return Dict(
                :Je => Je,
                :times => times,
                :V => V,
                :V2 => V2,
                :psi_avg => psi_avg,
                :T_avg => T_avg,
                :psi => psi,
                :T => T_snap,
                :skip_idx => 1,
                :V_avg => mean(V),
                :V2_avg => mean(V2),
                :psi_avg_mean => mean(psi_avg),
                :T_avg_mean => mean(T_avg),
                :V_t => [(times[i], V[i]) for i in eachindex(times)]
            )
        end
    end

    println("\n  Loading data from H5 files...")
    data_up = filter(!isnothing, [load_data_with_skip(r[:h5_file]) for r in results_up])
    data_down = filter(!isnothing, [load_data_with_skip(r[:h5_file]) for r in results_down])

    if isempty(data_up) || isempty(data_down)
        println("  Error: No data loaded!")
        return
    end

    # Extract summary
    V_up = [d[:V_avg] for d in data_up]
    V_down = [d[:V_avg] for d in data_down]
    V2_up = [d[:V2_avg] for d in data_up]
    V2_down = [d[:V2_avg] for d in data_down]
    psi_up = [d[:psi_avg_mean] for d in data_up]
    psi_down = [d[:psi_avg_mean] for d in data_down]
    T_up = [d[:T_avg_mean] for d in data_up]
    T_down = [d[:T_avg_mean] for d in data_down]
    Je_up = [d[:Je] for d in data_up]
    Je_down = [d[:Je] for d in data_down]

    println("  Loaded $(length(data_up)) up + $(length(data_down)) down points")

    # I-V curve (full domain)
    p_iv = plot(xlabel="Je", ylabel="V", title="I-V Characteristic",
                legend=:topleft, size=(700, 500))
    plot!(p_iv, Je_up, V_up, label="Up", color=:blue, lw=2, marker=:circle, ms=4)
    plot!(p_iv, Je_down, V_down, label="Down", color=:red, lw=2, marker=:square, ms=4)
    savefig(p_iv, joinpath(output_dir, "IV_curve.png"))
    println("  Saved: IV_curve.png")

    # I-V curve (hole region only)
    p_iv2 = plot(xlabel="Je", ylabel="V₂ (hole)", title="I-V Characteristic (hole region)",
                 legend=:topleft, size=(700, 500))
    plot!(p_iv2, Je_up, V2_up, label="Up", color=:blue, lw=2, marker=:circle, ms=4)
    plot!(p_iv2, Je_down, V2_down, label="Down", color=:red, lw=2, marker=:square, ms=4)
    savefig(p_iv2, joinpath(output_dir, "IV_curve_hole.png"))
    println("  Saved: IV_curve_hole.png")

    # Order parameter
    p_psi = plot(xlabel="Je", ylabel="<|ψ|>", title="Order Parameter",
                 legend=:topright, size=(700, 500))
    plot!(p_psi, Je_up, psi_up, label="Up", color=:blue, lw=2, marker=:circle, ms=4)
    plot!(p_psi, Je_down, psi_down, label="Down", color=:red, lw=2, marker=:square, ms=4)
    savefig(p_psi, joinpath(output_dir, "psi_vs_current.png"))
    println("  Saved: psi_vs_current.png")

    # Temperature
    p_T = plot(xlabel="Je", ylabel="<T>", title="Temperature",
               legend=:topleft, size=(700, 500))
    plot!(p_T, Je_up, T_up, label="Up", color=:blue, lw=2, marker=:circle, ms=4)
    plot!(p_T, Je_down, T_down, label="Down", color=:red, lw=2, marker=:square, ms=4)
    savefig(p_T, joinpath(output_dir, "T_vs_current.png"))
    println("  Saved: T_vs_current.png")

    # Eta distribution
    eta = compute_eta_distribution(p)
    p_eta = heatmap(x, y, eta[2:p.Nx, 2:p.Ny]',
                    xlabel="x", ylabel="y", title="eta distribution",
                    color=:viridis, aspect_ratio=:equal, size=(550, 500))
    savefig(p_eta, joinpath(output_dir, "eta_distribution.png"))
    println("  Saved: eta_distribution.png")

    # Tc distribution
    Tc = compute_Tc_distribution(p)
    p_Tc = heatmap(x, y, Tc[2:p.Nx, 2:p.Ny]',
                   xlabel="x", ylabel="y", title="Tc distribution",
                   color=:coolwarm, aspect_ratio=:equal, size=(550, 500))
    savefig(p_Tc, joinpath(output_dir, "Tc_distribution.png"))
    println("  Saved: Tc_distribution.png")

    # Find T_max for color scale
    T_max_all = 0.1
    for d in vcat(data_up, data_down)
        T_max_all = max(T_max_all, maximum(d[:T]))
    end

    # Find Je_max for 4-window animation
    Je_max = maximum(vcat(Je_up, Je_down))

    # Calculate V_max_all for 4-window animation
    V_max_all = max(maximum(V_up), maximum(V_down))

    # 4-window animation generation - commented out for nofastscan mode (Task 8)
    # futures = generate_parallel_4window_animation(output_dir, data_up, data_down,
    #                                              J_values, config.skip_ratio,
    #                                              x, y, T_max_all, V_max_all, Je_max,
    #                                              p.Nx, p.Ny)
    # n_frames = length(futures)
    # merge_animations_with_ffmpeg(output_dir, n_frames)

    open(joinpath(output_dir, "parameters.txt"), "w") do f
        println(f, "TDGL + Thermal FastScan Sweep (Standalone V2)")
        println(f, "Date: $(now())")
        println(f, "Workers: $(nworkers())")
        println(f, "Grid: $(p.Nx) x $(p.Ny)")
        println(f, "Tc1=$(p.Tc1), Tc2=$(p.Tc2), Tc3=$(p.Tc3), Tc1gap=$(p.Tc1gap), Tc3gap=$(p.Tc3gap)")
        println(f, "eta_env=$(p.eta_env), eta_hole=$(p.eta_hole), holewidth=$(p.holewidth)")
        println(f, "Jpeak=$(config.Jpeak), n_points=$(config.n_current_points)")
        println(f, "input_folder=$(isnothing(config.input_folder) ? "none" : config.input_folder)")
        println(f, "stable_time=$(config.stable_time), dt_snapshots=$(config.dt_snapshots), skip_ratio=$(config.skip_ratio)")
    end
    println("  Saved: parameters.txt")

    println("\n" * "=" ^ 60)
    println("Results saved to: $(abspath(output_dir))/")
    println("=" ^ 60)
end

# ============================================================================
# Main
# ============================================================================
function main()
    println("=" ^ 60)
    println("TDGL + Thermal No-FastScan Sweep")
    println("=" ^ 60)
    # Headless rendering: Set non-interactive GR driver for SLURM environments
    # This fixes GKS/Qt display errors when running without X11 display
    ENV["GKSwstype"] = "100"
    println("Date: $(now())")
    println()

    # Parse command line arguments
    # Usage:
    #   julia script.jl config.yaml                           - fresh run from config
    #   julia script.jl --from target_folder/                  - continuation run (skip_ratio=0.5 default)
    #   julia script.jl --from target_folder/ --skip_ratio 0.7 - continuation with custom skip_ratio
    #   julia script.jl --from target_folder/ --stable 500    - continuation with custom stable_time

    if length(ARGS) == 0
        error("Usage: julia script.jl config.yaml OR julia script.jl --from target_folder/ [--skip_ratio VALUE] [--stable VALUE]")
    end

    first_arg = ARGS[1]

    if first_arg == "--from"
        # Continuation mode: --from target_folder/
        if length(ARGS) < 2
            error("--from requires a folder path: julia script.jl --from target_folder/")
        end
        input_folder_from_cmd = ARGS[2]
        println("Continuation mode: reading from folder: $input_folder_from_cmd")

        # Parse optional --skip_ratio and --stable arguments
        skip_ratio_override = nothing
        stable_time_override = nothing
        i = 3
        while i <= length(ARGS)
            if ARGS[i] == "--skip_ratio" && i + 1 <= length(ARGS)
                skip_ratio_override = parse(Float64, ARGS[i+1])
                println("Override skip_ratio: $skip_ratio_override")
                i += 2
            elseif ARGS[i] == "--stable" && i + 1 <= length(ARGS)
                stable_time_override = parse(Float64, ARGS[i+1])
                println("Override stable_time: $stable_time_override")
                i += 2
            else
                i += 1
            end
        end

        # Load config from the continuation folder if available
        continuation_config_path = joinpath(input_folder_from_cmd, "config.yaml")
        if isfile(continuation_config_path)
            println("Loading config from continuation folder: $continuation_config_path")
            cfg = load_config_yaml(continuation_config_path)
        else
            println("Warning: No config.yaml found in continuation folder, using default parameters")
            cfg = nothing
        end
    else
        # Fresh run: config file
        config_path = first_arg
        println("Fresh run mode")
        println("Config path: $config_path")
        input_folder_from_cmd = nothing
        skip_ratio_override = nothing
        stable_time_override = nothing

        if isfile(config_path)
            println("Config file: FOUND")
            cfg = load_config_yaml(config_path)
        else
            println("Config file: NOT FOUND (using default parameters)")
            cfg = nothing
        end
    end

    p = build_params_from_config(cfg)
    config = build_sweep_config(cfg; input_folder_from_cmd=input_folder_from_cmd,
                               skip_ratio_override=skip_ratio_override,
                               stable_time_override)
    continuation_up_points = NamedTuple[]
    continuation_down_points = NamedTuple[]
    if !isnothing(config.input_folder)
        continuation_up_points, continuation_down_points = discover_input_folder_points(config.input_folder)
        effective_points = vcat(continuation_up_points, continuation_down_points)
        config.Jpeak = maximum(point.Je for point in effective_points)
        config.n_current_points = length(unique(point.Je for point in effective_points))
        config.run_up = !isempty(continuation_up_points)
        config.run_down = !isempty(continuation_down_points)
    end

    Lx = p.Nx * p.hx
    Ly = p.Ny * p.hy
    println("Grid: $(p.Nx) x $(p.Ny), domain: [-$(Lx/2), $(Lx/2)] x [-$(Ly/2), $(Ly/2)]")
    println("Tc: Tc1=$(p.Tc1), Tc2=$(p.Tc2), Tc3=$(p.Tc3), Tc1gap=$(p.Tc1gap), Tc3gap=$(p.Tc3gap)")
    if p.Tc1 == p.Tc2 == p.Tc3
        println("    Tc is UNIFORM = $(p.Tc1)")
    else
        y1_threshold = -Ly/2 + Ly * p.Tc1gap
        y3_threshold = Ly/2 - Ly * p.Tc3gap
        println("    Tc1 region: y < $(y1_threshold), Tc2 region: $(y1_threshold) <= y < $(y3_threshold), Tc3 region: y >= $(y3_threshold)")
    end
    println("eta: env=$(p.eta_env), hole=$(p.eta_hole), width=$(p.holewidth)")
    println("Jpeak=$(config.Jpeak), n_points=$(config.n_current_points)")
    println("input_folder=$(isnothing(config.input_folder) ? "none" : config.input_folder)")
    println()

    # Create output directory (always create new timestamped directory for consistent structure)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    output_dir = "sweep_nofastscan_$timestamp"
    mkpath(output_dir)
    println("Output directory: $output_dir")
    println("OUTPUT_DIR:$output_dir\n")

    # Save config.yaml and refine.yaml to output directory
    save_config_yaml(joinpath(output_dir, "config.yaml"), p, config)
    save_refine_yaml(joinpath(output_dir, "refine.yaml"), config)
    println("Saved: config.yaml, refine.yaml\n")

    # Create J values for sweep
    J_values = if isnothing(config.input_folder)
        collect(range(0, config.Jpeak, length=config.n_current_points))
    else
        sort(unique([point.Je for point in vcat(continuation_up_points, continuation_down_points)]))
    end

    # Warm up workers with precompilation
    # warmup_workers(p)  # DISABLED: Causing deadlock in SLURM with multiple workers

    if isnothing(config.input_folder) ? config.run_up : !isempty(continuation_up_points)
        println("\n" * "=" ^ 60)
        if isnothing(config.input_folder)
            println("SWEEP UP (0 → $(config.Jpeak))")
            println("  Initial: ψ=$(config.psi_up_init), T=$(config.T_initial_up)")
            results_up = pmap(J -> simulate_point(J, p, config, "up", output_dir), J_values)
        else
            println("SWEEP UP (continuation from input_folder)")
            println("  Points: $(length(continuation_up_points))")
            results_up = pmap(point -> simulate_point(point.Je, p, config, "up", output_dir;
                                                      initial_state_override=load_continuation_state(point.file)),
                              continuation_up_points)
        end
        println("=" ^ 60)

        Je_up = [r[:J] for r in results_up]
        V_up = [r[:V] for r in results_up]
        psi_up = [r[:psi] for r in results_up]
        T_up = [r[:T] for r in results_up]
    else
        println("\n" * "=" ^ 60)
        println("SWEEP UP skipped by config")
        println("=" ^ 60)
        results_up = Dict[]
        Je_up = Float64[]
        V_up = Float64[]
        psi_up = Array{Float64,3}[]
        T_up = Array{Float64,3}[]
    end

    if isnothing(config.input_folder) ? config.run_down : !isempty(continuation_down_points)
        println("\n" * "=" ^ 60)
        if isnothing(config.input_folder)
            println("SWEEP DOWN ($(config.Jpeak) → 0)")
            println("  Initial: ψ=$(config.psi_down_init), T=$(config.T_initial_down)")
            results_down = pmap(J -> simulate_point(J, p, config, "down", output_dir), reverse(J_values))
        else
            println("SWEEP DOWN (continuation from input_folder)")
            println("  Points: $(length(continuation_down_points))")
            results_down = pmap(point -> simulate_point(point.Je, p, config, "down", output_dir;
                                                        initial_state_override=load_continuation_state(point.file)),
                                continuation_down_points)
        end
        println("=" ^ 60)

        Je_down = [r[:J] for r in results_down]
        V_down = [r[:V] for r in results_down]
        psi_down = [r[:psi] for r in results_down]
        T_down = [r[:T] for r in results_down]
    else
        println("\n" * "=" ^ 60)
        println("SWEEP DOWN skipped by config")
        println("=" ^ 60)
        results_down = Dict[]
        Je_down = Float64[]
        V_down = Float64[]
        psi_down = Array{Float64,3}[]
        T_down = Array{Float64,3}[]
    end

    # Step 3: Generate outputs
    generate_outputs_nofastscan(output_dir, J_values, Je_up, V_up, psi_up, T_up,
                               Je_down, V_down, psi_down, T_down, p, config)

    return output_dir
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && get(ENV, "NOFASTSCAN_SKIP_MAIN", "0") != "1"
    output_dir = main()
    println("OUTPUT_DIR:$output_dir")
end
