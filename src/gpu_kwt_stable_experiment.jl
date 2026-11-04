using CUDA

@inline function kwt_old32(z::ComplexF32, r::ComplexF32, g::Float32, u::Float32)
    d=1f0+g*g*abs2(z); v=r/(u/sqrt(d)); c=real(conj(z)*v)/d
    v-g*g*c*z
end
@inline function kwt_stable32(z::ComplexF32, r::ComplexF32, g::Float32, u::Float32)
    d=1f0+g*g*abs2(z)
    (r+g*g*(im*z)*imag(conj(z)*r))/(u*sqrt(d))
end
@inline function kwt_promoted64(z::ComplexF32, r::ComplexF32, g::Float32, u::Float32)
    zz=ComplexF64(z); rr=ComplexF64(r); gg=Float64(g); uu=Float64(u); d=1+gg*gg*abs2(zz)
    ComplexF32((rr+gg*gg*(im*zz)*imag(conj(zz)*rr))/(uu*sqrt(d)))
end

function _k_dpsi_kwt!(dpsi,rhs,psi,ax,ay,temp,ux,uy,tc,s,::Val{Mode}) where {Mode}
    q=(CUDA.blockIdx().x-1)*CUDA.blockDim().x+CUDA.threadIdx().x; b=CUDA.blockIdx().y; nx=Int(s.nx); ny=Int(s.ny)
    if q <= (nx+1)*(ny+1)
        i=1+mod(q-1,nx+1); j=1+(q-1)÷(nx+1)
        if i>1 && i<=nx && j>1 && j<=ny
            z=psi[q,b]; r=(conj(ux[_IX(i-1,j,nx),b])*psi[_I(i-1,j,nx),b]-2z+ux[_IX(i,j,nx),b]*psi[_I(i+1,j,nx),b])/(s.hx*s.hx)
            r+=(conj(uy[_IY(i,j-1,nx),b])*psi[_I(i,j-1,nx),b]-2z+uy[_IY(i,j,nx),b]*psi[_I(i,j+1,nx),b])/(s.hy*s.hy)
            r+=(tc[q,b]-temp[q,b]-abs2(z))*z; rhs[q,b]=r
            dpsi[q,b]=Mode===:stable32 ? kwt_stable32(z,r,s.gamma,s.u) : kwt_promoted64(z,r,s.gamma,s.u)
        else dpsi[q,b]=0; rhs[q,b]=0 end
    end
    return
end

function gpu_full_curl32_kwt_rhs!(du::CuArray{Complex{T},2},u::CuArray{Complex{T},2},c::GPUFullCurl32Cache,t;mode::Symbol=:stable32) where {T}
    T in (Float32,Float64) || throw(ArgumentError("supported integration state types are Float32 and Float64"))
    mode in (:stable32,:promoted64) || throw(ArgumentError("mode must be :stable32 or :promoted64"))
    size(u)==size(du) || throw(DimensionMismatch("u and du must have identical packed shapes")); size(u,2)==length(c.currents) || throw(DimensionMismatch("batch count differs from cache currents"))
    CUDA.allowscalar(false); s=c.s; b=size(u,2); np=(Int(s.nx)+1)*(Int(s.ny)+1); na=Int(s.nx)*(Int(s.ny)+1); nb=(Int(s.nx)+1)*Int(s.ny); packed=2*np+na+nb; th=256
    @cuda threads=th blocks=(cld(packed,th),b) _k_unpack!(du,u,c.psi,c.ax,c.ay,c.temp,c.ux,c.uy,s)
    @cuda threads=th blocks=(cld(max(size(c.ux,1),size(c.uy,1)),th),b) _k_links!(c.psi,c.ax,c.ay,c.temp,c.ux,c.uy,s)
    @cuda threads=128 blocks=(cld(max(Int(s.nx),Int(s.ny)+1),128),b) _k_bc_x!(c.psi,c.ax,c.ay,c.temp,c.currents,s)
    @cuda threads=128 blocks=(cld(Int(s.nx)+1,128),b) _k_bc_y!(c.psi,c.temp,c.uy,s)
    @cuda threads=th blocks=(cld(np,th),b) _k_dpsi_kwt!(c.dpsi,c.rhspsi,c.psi,c.ax,c.ay,c.temp,c.ux,c.uy,c.tc,s,Val(mode))
    @cuda threads=th blocks=(cld(max(size(c.jsx,1),size(c.jsy,1)),th),b) _k_current!(c.jsx,c.jsy,c.psi,c.ux,c.uy,s)
    @cuda threads=th blocks=(cld(max(size(c.dax,1),size(c.day,1)),th),b) _k_vector!(c.dax,c.day,c.jsx,c.jsy,c.ax,c.ay,c.sigma,s)
    @cuda threads=th blocks=(cld(np,th),b) _k_heat!(c.dt,c.w,c.dpsi,c.dax,c.day,c.psi,c.temp,c.eta,c.sigma,s)
    @cuda threads=th blocks=(cld(packed,th),b) _k_pack!(du,c.dpsi,c.dax,c.day,c.dt,s)
    du
end

