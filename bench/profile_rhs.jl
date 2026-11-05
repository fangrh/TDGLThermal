using JSON3, Random, Statistics, Profile
include(joinpath(@__DIR__, "fixtures.jl"))

function measure(label, f; count=2000)
    f()
    allocation = @allocated f()
    elapsed = @elapsed for _ in 1:count
        f()
    end
    (; label, seconds_per_call=elapsed/count, allocated_bytes=allocation, count)
end

function main()
    config = joinpath(@__DIR__, "..", "config.yaml")
    case = make_case(config; Je=0.18, noise=0.0, seed=20261104,
                     preparation=:cool)
    p = case.params
    cache = TDGLThermalCache(p)
    u = copy(case.u0)
    du = similar(u)
    fdm_rhs_compatible!(du,u,cache,0.0)
    rows = [
        measure("reference_rhs", () -> fdm_rhs_thermal!(du,u,cache,0.0)),
        measure("compatible_rhs", () -> fdm_rhs_compatible!(du,u,cache,0.0)),
        measure("reference_unpack_and_links", () -> vector_to_state!(cache.state,u,p)),
        measure("compatible_unpack_and_links", () -> vector_to_state_compatible!(cache.state,u,p)),
        measure("links", () -> update_link_variables!(cache.state,p)),
        measure("compatible_links", () -> update_link_variables_compatible!(cache.state,p)),
        measure("boundaries", () -> apply_boundary_conditions!(cache.state,p)),
        measure("dpsi", () -> compute_dpsi_dt!(cache.dpsi,cache.RHS_psi,cache.state,p,cache.Tc)),
        measure("supercurrent", () -> compute_supercurrent!(cache.Jsx,cache.Jsy,cache.state,p)),
        measure("full_curl", () -> compute_dA_dt!(cache.dAx,cache.dAy,cache.state,p,cache.Jsx,cache.Jsy,cache.sigma)),
        measure("heat_production", () -> compute_heat_production!(cache.W_total,cache.state,p,cache.dpsi,cache.dAx,cache.dAy,cache.sigma)),
        measure("thermal_diffusion", () -> compute_dT_dt!(cache.dT,cache.state,p,cache.W_total,cache.eta))
    ]
    Profile.clear()
    @profile for _ in 1:5000
        fdm_rhs_compatible!(du,u,cache,0.0)
    end
    output = length(ARGS)>0 ? ARGS[1] : "rhs_profile.json"
    open(output,"w") do io
        write(io,JSON3.write((julia=string(VERSION),grid=(p.Nx,p.Ny),noise=0.0,rows)))
    end
    open(output*".txt","w") do io
        Profile.print(io; format=:flat, sortedby=:count)
    end
    println(JSON3.write(rows))
end
main()
