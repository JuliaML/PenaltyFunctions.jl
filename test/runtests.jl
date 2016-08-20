module Tests
using LearnBase, Penalties
using Base.Test

@testset "L1Penalty" begin
    p = L1Penalty(.1)
    β = randn(10)
    storage = zeros(10)
    @test value(p, β) ≈ .1 * sumabs(β)
    @test deriv(p, β[1]) ≈ .1 * sign(β[1])
    grad!(storage, p, β)
    @test storage ≈ .1 * sign.(β)
    @test prox(p, β[1]) ≈ sign(β[1]) * max(0., abs(β[1]) - .1)
end

@testset "L2Penalty" begin
    p = L2Penalty(.1)
    β = randn(10)
    storage = zeros(10)
    @test value(p, β) ≈ .5 * .1 * sumabs2(β)
    @test deriv(p, β[1]) ≈ .1 * β[1]
    grad!(storage, p, β)
    @test storage ≈ .1 * β
    @test prox(p, β[1]) ≈ β[1] / (1.0 + 0.1)
end

@testset "Abstract methods" begin
    p = L1Penalty(.1)
    β = randn(10)
    βcopy = deepcopy(β)
    factor = rand(10)
    storage = zeros(10)

    @test value(p, β) ≈ p.λ * sumabs(β)
    @test value(p, β, factor) ≈ p.λ * sum(factor .* abs(β))

    grad!(storage, p, β)
    @test storage ≈ p.λ * sign(β)
    grad!(storage, p, β, factor)
    @test storage ≈ p.λ * sign(β) .* factor

    prox!(p, β)
    @test β ≈ Penalties.soft_thresh.(βcopy, p.λ)
    β = deepcopy(βcopy)
    prox!(p, β, factor)
    @test β ≈ [Penalties.soft_thresh(βcopy[i], p.λ * factor[i]) for i in 1:10]

end

end
