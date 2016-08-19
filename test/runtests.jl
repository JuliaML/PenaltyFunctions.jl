module Tests
using Penalties
using Base.Test




@testset "L1Penalty" begin
    p = L1Penalty()
    β = randn(10)
    @test value(p, .1, β) ≈ .1 * sumabs(β)
    @test deriv(p, .1, β[1]) ≈ .1 * sign(β[1])
    @test grad(p, .1, β) ≈ .1 * sign.(β)
end
end
