using Test

include(joinpath(@__DIR__, "..", "bench", "ndr_receipt_hashes.jl"))

const H = repeat("a", 64)
const G = repeat("b", 64)

@testset "NDR receipt hash metadata" begin
    @test checked_preceding_sha(nothing, nothing, nothing) === nothing
    @test checked_preceding_sha("initial.h5", H, H) == H
    @test_throws ErrorException checked_preceding_sha("initial.h5", "bad", H)
    @test_throws ErrorException checked_preceding_sha("initial.h5", H, G)
    @test_throws ErrorException checked_preceding_sha("initial.h5", uppercase(H), H)
    @test_throws ErrorException checked_preceding_sha(nothing, H, nothing)
    @test_throws ErrorException checked_preceding_sha(nothing, nothing, H)
    @test_throws ErrorException checked_preceding_sha("initial.h5", nothing, H)
    @test_throws ErrorException checked_preceding_sha("initial.h5", H, nothing)
    @test_throws ErrorException checked_preceding_sha("initial.h5", H, "bad")
    @test checked_output_sha(H) == H
    @test_throws ErrorException checked_output_sha("missing")
    @test_throws ErrorException checked_output_sha(uppercase(H))
    @test_throws ErrorException checked_output_sha(nothing)
end
