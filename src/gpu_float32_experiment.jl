# Experimental deterministic Float32/mixed-precision RHS.  This is not a
# production backend and intentionally keeps the validated CUDA kernel code
# in gpu_full_curl.jl as its single stencil implementation.
using CUDA

struct GPUFullCurl32Scalars
    nx::Int32; ny::Int32; hx::Float32; hy::Float32
    kappa::Float32; u::Float32; gamma::Float32
    ceff::Float32; keff::Float32; t0::Float32
    eta_env::Float32; eta_hole::Float32; holewidth::Float32
    he::Float32; coverage::Float32
    tc1::Float32; tc2::Float32; tc3::Float32; tc1gap::Float32; tc3gap::Float32
    thermal::Int32
end

struct GPUFullCurl32Cache
    s::GPUFullCurl32Scalars
    currents::CuArray{Float32,1}
    sigma::CuArray{Float32,2}; eta::CuArray{Float32,2}; tc::CuArray{Float32,2}
    psi::CuArray{ComplexF32,2}; ax::CuArray{Float32,2}; ay::CuArray{Float32,2}; temp::CuArray{Float32,2}
    ux::CuArray{ComplexF32,2}; uy::CuArray{ComplexF32,2}
    dpsi::CuArray{ComplexF32,2}; rhspsi::CuArray{ComplexF32,2}
    dax::CuArray{Float32,2}; day::CuArray{Float32,2}; dt::CuArray{Float32,2}
    jsx::CuArray{Float32,2}; jsy::CuArray{Float32,2}; w::CuArray{Float32,2}
end

function GPUFullCurl32Cache(p::TDGLThermalParams, currents::Vector{Float64})
    _gpu_supported(p)
    isempty(currents) && throw(ArgumentError("currents must not be empty"))
    nx,ny,b = p.Nx,p.Ny,length(currents)
    np=(nx+1)*(ny+1); na=nx*(ny+1); nb=(nx+1)*ny
    s=GPUFullCurl32Scalars(Int32(nx),Int32(ny),Float32(p.hx),Float32(p.hy),Float32(p.kappa),Float32(p.u),Float32(p.gamma),Float32(p.C_eff),Float32(p.k_eff),Float32(p.T0),Float32(p.eta_env),Float32(p.eta_hole),Float32(p.holewidth),Float32(p.He),Float32(p.bc_y_coverage),Float32(p.Tc1),Float32(p.Tc2),Float32(p.Tc3),Float32(p.Tc1gap),Float32(p.Tc3gap),p.thermal_enabled ? Int32(1) : Int32(0))
    sigma,eta,tc=_gpu_maps(p)
    zd(n,T)=CUDA.zeros(T,n,b)
    GPUFullCurl32Cache(s,CuArray(Float32.(currents)),CuArray(Float32.(repeat(vec(sigma),1,b))),CuArray(Float32.(repeat(vec(eta),1,b))),CuArray(Float32.(repeat(vec(tc),1,b))),zd(np,ComplexF32),zd(na,Float32),zd(nb,Float32),zd(np,Float32),zd(na,ComplexF32),zd(nb,ComplexF32),zd(np,ComplexF32),zd(np,ComplexF32),zd(na,Float32),zd(nb,Float32),zd(np,Float32),zd(na,Float32),zd(nb,Float32),zd(np,Float32))
end

function gpu_full_curl32_rhs!(du::CuArray{ComplexF32,2}, u::CuArray{ComplexF32,2}, c::GPUFullCurl32Cache, t)
    size(u)==size(du) || throw(DimensionMismatch("u and du must have identical packed shapes"))
    size(u,2)==length(c.currents) || throw(DimensionMismatch("batch count differs from cache currents"))
    CUDA.allowscalar(false); s=c.s; b=size(u,2); np=(Int(s.nx)+1)*(Int(s.ny)+1); na=Int(s.nx)*(Int(s.ny)+1); nb=(Int(s.nx)+1)*Int(s.ny); packed=2*np+na+nb; th=256
    @cuda threads=th blocks=(cld(packed,th),b) _k_unpack!(du,u,c.psi,c.ax,c.ay,c.temp,c.ux,c.uy,s)
    @cuda threads=th blocks=(cld(max(size(c.ux,1),size(c.uy,1)),th),b) _k_links!(c.psi,c.ax,c.ay,c.temp,c.ux,c.uy,s)
    @cuda threads=128 blocks=(cld(max(Int(s.nx),Int(s.ny)+1),128),b) _k_bc_x!(c.psi,c.ax,c.ay,c.temp,c.currents,s)
    @cuda threads=128 blocks=(cld(Int(s.nx)+1,128),b) _k_bc_y!(c.psi,c.temp,c.uy,s)
    @cuda threads=th blocks=(cld(np,th),b) _k_dpsi!(c.dpsi,c.rhspsi,c.psi,c.ax,c.ay,c.temp,c.ux,c.uy,c.tc,s)
    @cuda threads=th blocks=(cld(max(size(c.jsx,1),size(c.jsy,1)),th),b) _k_current!(c.jsx,c.jsy,c.psi,c.ux,c.uy,s)
    @cuda threads=th blocks=(cld(max(size(c.dax,1),size(c.day,1)),th),b) _k_vector!(c.dax,c.day,c.jsx,c.jsy,c.ax,c.ay,c.sigma,s)
    @cuda threads=th blocks=(cld(np,th),b) _k_heat!(c.dt,c.w,c.dpsi,c.dax,c.day,c.psi,c.temp,c.eta,c.sigma,s)
    @cuda threads=th blocks=(cld(packed,th),b) _k_pack!(du,c.dpsi,c.dax,c.day,c.dt,s)
    return du
end
