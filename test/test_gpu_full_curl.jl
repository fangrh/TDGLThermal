using Test
using Random

# The Linux controller runs this file with CUDA 13.4 and a real device.  On
# development hosts without CUDA, retain the test source and skip explicitly.
cuda_ok = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

if !cuda_ok
    @info "CUDA functional device unavailable; GPU full-curl tests skipped"
    @test true
else
    include(joinpath(@__DIR__, "..", "current_sweep_thermal_nofastscan.jl"))
    include(joinpath(@__DIR__, "..", "src", "gpu_full_curl.jl"))

    function fixture(p, seed)
        Random.seed!(seed)
        psi = complex.(0.3 .* randn(p.Nx+1,p.Ny+1), 0.3 .* randn(p.Nx+1,p.Ny+1))
        Ax = 0.1 .* randn(p.Nx,p.Ny+1); Ay = 0.1 .* randn(p.Nx+1,p.Ny)
        T = 0.2 .+ 0.2 .* rand(p.Nx+1,p.Ny+1)
        state_to_vector(TDGLThermalState(psi,Ax,Ay,T,zeros(ComplexF64,size(Ax)),zeros(ComplexF64,size(Ay))))
    end

    function check_case(p, currents; seed=20261104)
        us = [fixture(with_current(p,j), seed+i) for (i,j) in enumerate(currents)]
        u = hcat(us...); cpu = similar(u); du = similar(u); cpu_caches=TDGLThermalCache[]
        for (b,j) in enumerate(currents)
            pc = with_current(p,j); cc = TDGLThermalCache(pc)
            fdm_rhs_thermal!(view(cpu,:,b), view(u,:,b), cc, 0.0)
            push!(cpu_caches,cc)
        end
        g = GPUFullCurlCache(p, currents); gu = CUDA.CuArray(u); gd = CUDA.zeros(ComplexF64,size(u))
        before = copy(Array(gu)); gpu_full_curl_rhs!(gd,gu,g,0.0); CUDA.synchronize()
        @test Array(gu) == before
        # Directly inspect the reconstructed fields and sampled links too:
        # zero derivative boundary rows must not hide a corner/order mismatch.
        for (device_field,host_name) in ((g.psi,:psi),(g.ax,:Ax),(g.ay,:Ay),(g.temp,:T),(g.ux,:Ux),(g.uy,:Uy))
            host=Array(device_field)
            for b in axes(u,2)
                expected=vec(getproperty(cpu_caches[b].state,host_name))
                @test maximum(abs,host[:,b]-expected)/(1+maximum(abs,expected)) <= 1e-11
            end
        end
        got = Array(gd); nx=p.Nx; ny=p.Ny
        np=(nx+1)*(ny+1); na=nx*(ny+1); nb=(nx+1)*ny
        ranges=(1:np, np+1:np+na, np+na+1:np+na+nb, np+na+nb+1:np+na+nb+np)
        names=("psi","Ax","Ay","T")
        for (name,r) in zip(names,ranges), b in axes(u,2)
            err=maximum(abs, got[r,b]-cpu[r,b])/(1+maximum(abs,cpu[r,b]))
            @info "GPU field error" field=name batch=b normalized_error=err cpu_max=maximum(abs,cpu[r,b]) gpu_max=maximum(abs,got[r,b])
            @test err <= 1e-11
        end
    end

    @testset "deterministic GPU corrected full-curl RHS" begin
        p = TDGLThermalParams(Nx=7, Ny=5, hx=0.5, hy=0.75, kappa=2.0, gamma=4.0,
            sigma1=0.7, sigma2=1.0, sigma3=1.4, Tc1gap=0.2, Tc3gap=0.2,
            Je=0.12, He=0.03, bc_x=:periodic, bc_y=:neumann,
            bc_y_coverage=0.88, loop=false, field_side=:both, psi_y_zero=false,
            epsilon_type=:linear, noise_strength=0.0)
        check_case(p, [0.06,0.18,0.27])
        p60 = TDGLThermalParams(Nx=60, Ny=30, hx=0.5, hy=0.5, kappa=2.0, gamma=40.0,
            sigma1=1.0, sigma2=1.0, sigma3=1.0, Tc1=1.0, Tc2=1.0, Tc3=1.0,
            Tc1gap=0.2, Tc3gap=0.0, C_eff=1.0, k_eff=5.0, T0=0.18,
            eta_env=10.0, eta_hole=0.1, holewidth=20.0, bc_x=:periodic,
            bc_y=:neumann, bc_y_coverage=0.88, loop=false, field_side=:both,
            psi_y_zero=false, thermal_enabled=true, epsilon_type=:linear,
            noise_strength=0.0)
        check_case(p60, [0.06,0.18,0.27])
    end

    @testset "unsupported GPU RHS options reject explicitly" begin
        base = TDGLThermalParams(Nx=7,Ny=5,bc_x=:periodic,bc_y=:neumann,noise_strength=0.0)
        @test_throws ArgumentError GPUFullCurlCache(TDGLThermalParams(Nx=7,Ny=5,bc_x=:neumann), [0.1])
        @test_throws ArgumentError GPUFullCurlCache(TDGLThermalParams(Nx=7,Ny=5,noise_strength=0.001), [0.1])
        @test_throws ArgumentError GPUFullCurlCache(TDGLThermalParams(Nx=7,Ny=5,loop=true), [0.1])
        @test_throws ArgumentError GPUFullCurlCache(TDGLThermalParams(Nx=7,Ny=5,field_side=:upper), [0.1])
        @test_throws ArgumentError GPUFullCurlCache(TDGLThermalParams(Nx=7,Ny=5,epsilon_type=:bcs), [0.1])
        @test_throws ArgumentError GPUFullCurlCache(TDGLThermalParams(Nx=2,Ny=5), [0.1])
        @test base.noise_strength == 0.0
    end
end
