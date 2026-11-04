using Test
using Random
cuda_ok = try; @eval using CUDA; CUDA.functional(); catch; false end

if !cuda_ok
    @info "CUDA functional device unavailable; Float32 experiment skipped"
    @test true
else
    include(joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl"))
    include(joinpath(@__DIR__, "..", "src", "gpu_full_curl.jl"))
    include(joinpath(@__DIR__, "..", "src", "gpu_float32_experiment.jl"))

    function fixture(p, seed)
        Random.seed!(seed); sh=(p.Nx+1,p.Ny+1)
        psi=complex.(0.3 .* randn(sh),0.3 .* randn(sh)); Ax=0.1 .* randn(p.Nx,p.Ny+1)
        Ay=0.1 .* randn(p.Nx+1,p.Ny); T=0.2 .+ 0.2 .* rand(sh...)
        state_to_vector(TDGLThermalState(psi,Ax,Ay,T,zeros(ComplexF64,size(Ax)),zeros(ComplexF64,size(Ay))))
    end
    function run_case(p, currents)
        u64=hcat([fixture(with_current(p,j),20261104+b) for (b,j) in enumerate(currents)]...)
        cpu=zeros(ComplexF64,size(u64)); for (b,j) in enumerate(currents); fdm_rhs_thermal!(view(cpu,:,b),view(u64,:,b),TDGLThermalCache(with_current(p,j)),0.0); end
        u32=CUDA.CuArray(ComplexF32.(u64)); d32=CUDA.zeros(ComplexF32,size(u32)); before=copy(Array(u32)); c32=GPUFullCurl32Cache(p,currents)
        gpu_full_curl32_rhs!(d32,u32,c32,0.0); CUDA.synchronize(); @test Array(u32)==before
        got=ComplexF64.(Array(d32)); nx,ny=p.Nx,p.Ny; np=(nx+1)*(ny+1); na=nx*(ny+1); nb=(nx+1)*ny
        ranges=("psi"=>1:np,"Ax"=>np+1:np+na,"Ay"=>np+na+1:np+na+nb,"T"=>np+na+nb+1:np+na+nb+np)
        for (name,r) in ranges
            e=maximum(abs,got[r,:]-cpu[r,:])/(1+maximum(abs,cpu[r,:]))
            @info "Float32 field error" field=name normalized_error=e
            @test isfinite(e) && e<=1e-3 # Roundoff screen only, not a trajectory accuracy gate.
        end
    end
    @testset "Float32 full-curl feasibility" begin
        p=TDGLThermalParams(Nx=7,Ny=5,hx=.5,hy=.75,kappa=2.,gamma=4.,sigma1=.7,sigma2=1.,sigma3=1.4,Tc1gap=.2,Tc3gap=.2,Je=.12,He=.03,bc_x=:periodic,bc_y=:neumann,bc_y_coverage=.88,field_side=:both,epsilon_type=:linear,noise_strength=0.)
        run_case(p,[.06,.18,.27])
        p60=TDGLThermalParams(Nx=60,Ny=30,hx=.5,hy=.5,kappa=2.,gamma=40.,sigma=1.,Tc1gap=.2,Tc3gap=0.,k_eff=5.,T0=.18,eta_env=10.,eta_hole=.1,holewidth=20.,bc_x=:periodic,bc_y=:neumann,bc_y_coverage=.88,field_side=:both,epsilon_type=:linear,noise_strength=0.)
        run_case(p60,[.06,.18,.27])
    end
end
