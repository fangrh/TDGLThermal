using Test,CUDA
include(joinpath(@__DIR__,"..","bench","benchmark_gpu_solve.jl"))
@testset "GPU per-trajectory adaptive norm" begin
    a=zeros(ComplexF64,17,3);a[:,2].=3+4im
    @test _batch_internalnorm(a,0.0)==5
    @test _batch_internalnorm(CuArray(a),0.0)==5
    @test _batch_internalnorm(3+4im,0.0)==5
    got=Array(_batch_internalnorm.(CuArray(a),0.0))
    @test got==abs.(a)
end
