"""Deterministic, opt-in coupled A/T IMEX experiment.

This file is deliberately independent of the production backend.  The spatial
operator is obtained by probing the actual boundary-conditioned kernels; no
stencil or boundary approximation is used here.
"""

using LinearAlgebra
using SparseArrays

mutable struct IsolatedSpatialCache
    params::TDGLThermalParams
    cache::TDGLThermalCache
    zero_u::Vector{ComplexF64}
    zero_out::Vector{ComplexF64}
end

function IsolatedSpatialCache(p::TDGLThermalParams)
    n = (p.Nx + 1) * (p.Ny + 1) + p.Nx * (p.Ny + 1) +
        (p.Nx + 1) * p.Ny + (p.Nx + 1) * (p.Ny + 1)
    IsolatedSpatialCache(p, TDGLThermalCache(p), zeros(ComplexF64, n),
                         zeros(ComplexF64, n))
end

"""Evaluate the affine spatial A/T map with ψ, supercurrent and heat removed."""
function isolated_spatial!(out::AbstractVector{ComplexF64}, u::AbstractVector,
                           c::IsolatedSpatialCache)
    p, k = c.params, c.cache
    length(out) == length(c.zero_u) || throw(DimensionMismatch("packed state length"))

    # This copy is complete on every call, so boundary reconstruction cannot
    # leak from a previous probe.  A/T are restricted to their physical real
    # subspace by the production unpacking convention.
    vector_to_state_compatible!(k.state, u, p)
    apply_boundary_conditions!(k.state, p)
    fill!(k.Jsx, 0.0)
    fill!(k.Jsy, 0.0)
    fill!(k.W_total, 0.0)
    compute_dA_dt!(k.dAx, k.dAy, k.state, p, k.Jsx, k.Jsy, k.sigma)
    if p.thermal_enabled
        compute_dT_dt!(k.dT, k.state, p, k.W_total, k.eta)
    else
        fill!(k.dT, 0.0)
    end

    nψ = (p.Nx + 1) * (p.Ny + 1)
    nAx = p.Nx * (p.Ny + 1)
    nAy = (p.Nx + 1) * p.Ny
    fill!(view(out, 1:nψ), 0.0)
    out[nψ + 1:nψ + nAx] .= ComplexF64.(vec(k.dAx))
    out[nψ + nAx + 1:nψ + nAx + nAy] .= ComplexF64.(vec(k.dAy))
    out[nψ + nAx + nAy + 1:end] .= ComplexF64.(vec(k.dT))
    return nothing
end

mutable struct SpatialSplit
    params::TDGLThermalParams
    cache::IsolatedSpatialCache
    L::SparseMatrixCSC{ComplexF64, Int}
    affine::Vector{ComplexF64}
end

"""Assemble ``G(u)=L*u+affine`` by exact finite-difference basis probes."""
function build_spatial_split(p::TDGLThermalParams; h::Float64=1.0)
    h > 0 || throw(ArgumentError("probe amplitude must be positive"))
    c = IsolatedSpatialCache(p)
    n = length(c.zero_u)
    g0 = similar(c.zero_out)
    isolated_spatial!(g0, c.zero_u, c)
    affine = copy(g0)
    rows = Int[]; cols = Int[]; vals = ComplexF64[]
    probe = zeros(ComplexF64, n)
    gp = similar(g0)
    for j in 1:n
        fill!(probe, 0.0)
        probe[j] = h
        isolated_spatial!(gp, probe, c)
        @inbounds for i in eachindex(gp)
            v = (gp[i] - g0[i]) / h
            if !iszero(v)
                push!(rows, i); push!(cols, j); push!(vals, v)
            end
        end
    end
    L = sparse(rows, cols, vals, n, n)
    return SpatialSplit(p, c, L, affine)
end

mutable struct ImexContext
    L::SparseMatrixCSC{ComplexF64, Int}
    affine::Vector{ComplexF64}
    reference_cache::TDGLThermalCache
    linear_work::Vector{ComplexF64}
end

function ImexContext(split::SpatialSplit;
                     reference_cache::TDGLThermalCache=TDGLThermalCache(split.params))
    ImexContext(split.L, split.affine, reference_cache,
                zeros(ComplexF64, size(split.L, 1)))
end

ImexContext(split::SpatialSplit, cache::TDGLThermalCache) =
    ImexContext(split.L, split.affine, cache, zeros(ComplexF64, size(split.L, 1)))

ImexContext(split::SpatialSplit, p::TDGLThermalParams) =
    ImexContext(split; reference_cache=TDGLThermalCache(p))

"""Explicit residual for ``F_reference(u)=L*u+N(u)``.

The full derivative-based heat production is evaluated by the reference RHS
before subtracting the linear spatial map; it is never reconstructed from a
residual derivative.
"""
function imex_residual!(du, u, ctx::ImexContext, t)
    fdm_rhs_compatible!(du, u, ctx.reference_cache, t)
    mul!(ctx.linear_work, ctx.L, u)
    @inbounds @simd for i in eachindex(du)
        du[i] -= ctx.linear_work[i]
    end
    return nothing
end

