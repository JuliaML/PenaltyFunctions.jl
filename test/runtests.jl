module Tests
using LearnBase, Penalties
using Base.Test




@testset "L1Penalty" begin
    p = L1Penalty(.1)
    β = randn(10)
    @test value(p, β) ≈ .1 * sumabs(β)
    @test deriv(p, β[1]) ≈ .1 * sign(β[1])
    @test grad(p, β) ≈ .1 * sign.(β)
end
end
