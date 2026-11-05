using Test
include(joinpath(@__DIR__,"..","bench","benchmark_gpu_precision_solve.jl"))
@testset "Float64 accumulation preserves the Float32 RHS work" begin
    p,currents,cases=_cases(joinpath(@__DIR__,"..","config.yaml"),3,60,30,:cool)
    hu=hcat([c.u0 for c in cases]...)
    u64=CuArray(hu);u32=CuArray(ComplexF32.(hu));d64=similar(u64);d32=similar(u32)
    a=GPUFullCurl32Cache(p,currents);b=GPUFullCurl32Cache(p,currents)
    for mode in (:stable32,:promoted64)
        gpu_full_curl32_kwt_rhs!(d64,u64,a,0.;mode=mode)
        gpu_full_curl32_kwt_rhs!(d32,u32,b,0.;mode=mode)
        CUDA.synchronize()
        @test Array(d64)==ComplexF64.(Array(d32))
        @test Array(u64)==hu
    end
end
