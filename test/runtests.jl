module Tests
using LearnBase, Penalties, Base.Test

@testset "Sanity Check" begin
    for p in [NoPenalty(), L1Penalty(.1), L2Penalty(.1),
              ElasticNetPenalty(.1, .5), SCADPenalty(.1, 3.7)]
        β = randn(5)
        w = rand(5)
        storage = zeros(5)

        # Evaluate on Number
        value(p, β[1])
        deriv(p, β[1])
        prox(p, β[1])

        # Evaluate on Number with scaled λ
        value(p, β[1], w[1])
        deriv(p, β[1], w[1])
        prox(p, β[1], w[1])

        # Evaluate on array
        value(p, β)
        grad!(storage, p, β)
        prox!(p, β)

        # Evaluate on array with scaled λ
        value(p, β, w[1])
        grad!(storage, p, β, w[1])
        prox!(p, β, w[1])

        # Evaluate on array with element-wise scaled λ
        value(p, β, w)
        grad!(storage, p, β, w)
        prox!(p, β, w)
    end
end

@testset "L1Penalty" begin
    p = L1Penalty(.1)
    β = randn(10)
    storage = zeros(10)
    @test value(p, β) ≈ .1 * sumabs(β)
    @test deriv(p, β[1]) ≈ .1 * sign(β[1])
    grad!(storage, p, β)
    @test storage ≈ .1 * map(sign, β)
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

@testset "ElasticNetPenalty" begin
    p = ElasticNetPenalty(.1, .7)
    β = randn(10)
    storage = zeros(10)
    @test value(p, β) ≈ .1 * (.7 * sumabs(β) + .3 * .5 * sumabs2(β))
    @test deriv(p, β[1]) ≈ .1 * .7 * sign(β[1]) + .1 * .3 * β[1]
    grad!(storage, p, β)
    @test storage ≈ .1 * .7 * map(sign, β) + .1 * .3 * β
    @test prox(p, β[1]) ≈ Penalties.soft_thresh(β[1], .07) / 1.03
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
    @test β ≈ map(x->Penalties.soft_thresh(x, p.λ), βcopy)
    β = deepcopy(βcopy)
    prox!(p, β, factor)
    @test β ≈ [Penalties.soft_thresh(βcopy[i], p.λ * factor[i]) for i in 1:10]
end

end
