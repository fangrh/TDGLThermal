"""Allocation-reduced CPU RHS compatible with the packed reference state."""

@inline function _copy_complex_field!(dest, u, offset::Int)
    @inbounds for k in eachindex(dest)
        dest[k] = u[offset + k]
    end
    return nothing
end

@inline function _copy_real_field!(dest, u, offset::Int)
    @inbounds for k in eachindex(dest)
        dest[k] = real(u[offset + k])
    end
    return nothing
end

@inline function _pack_field!(du, src, offset::Int)
    @inbounds for k in eachindex(src)
        du[offset + k] = src[k]
    end
    return nothing
end

"""Update link variables elementwise without broadcast temporaries or exp."""
function update_link_variables_compatible!(state, p::TDGLThermalParams)
    @inbounds for k in eachindex(state.Ux)
        state.Ux[k] = cis(-p.hx * state.Ax[k])
    end
    @inbounds for k in eachindex(state.Uy)
        state.Uy[k] = cis(-p.hy * state.Ay[k])
    end
    return nothing
end

"""Unpack a packed ComplexF64 state without slices or temporary arrays."""
function vector_to_state_compatible!(state, u, p::TDGLThermalParams)
    nx, ny = p.Nx, p.Ny
    len_psi = (nx + 1) * (ny + 1)
    len_Ax = nx * (ny + 1)
    len_Ay = (nx + 1) * ny
    _copy_complex_field!(state.psi, u, 0)
    _copy_real_field!(state.Ax, u, len_psi)
    _copy_real_field!(state.Ay, u, len_psi + len_Ax)
    _copy_real_field!(state.T, u, len_psi + len_Ax + len_Ay)
    update_link_variables_compatible!(state, p)
    return nothing
end

"""Compatible full-curl thermal RHS using the reference component kernels."""
function fdm_rhs_compatible!(du, u, cache::TDGLThermalCache, t)
    params = cache.params
    state = cache.state
    dpsi = cache.dpsi
    RHS_psi = cache.RHS_psi
    dAx = cache.dAx
    dAy = cache.dAy
    dT = cache.dT
    Jsx = cache.Jsx
    Jsy = cache.Jsy
    W_total = cache.W_total
    eta = cache.eta
    Tc = cache.Tc
    sigma = cache.sigma

    vector_to_state_compatible!(state, u, params)
    apply_boundary_conditions!(state, params)
    compute_dpsi_dt!(dpsi, RHS_psi, state, params, Tc)
    compute_supercurrent!(Jsx, Jsy, state, params)
    compute_dA_dt!(dAx, dAy, state, params, Jsx, Jsy, sigma)

    if params.thermal_enabled
        compute_heat_production!(W_total, state, params, dpsi, dAx, dAy, sigma)
        compute_dT_dt!(dT, state, params, W_total, eta)
    else
        fill!(dT, 0.0)
    end

    nx, ny = params.Nx, params.Ny
    len_psi = (nx + 1) * (ny + 1)
    len_Ax = nx * (ny + 1)
    len_Ay = (nx + 1) * ny
    _pack_field!(du, dpsi, 0)
    _pack_field!(du, dAx, len_psi)
    _pack_field!(du, dAy, len_psi + len_Ax)
    _pack_field!(du, dT, len_psi + len_Ax + len_Ay)
    return nothing
end

function resolve_rhs_backend(name::AbstractString=get(ENV, "TDGL_RHS_BACKEND", "reference"))
    name == "reference" && return fdm_rhs_thermal!
    name == "cpu_compatible" && return fdm_rhs_compatible!
    throw(ArgumentError("unknown TDGL_RHS_BACKEND: " * name))
end
