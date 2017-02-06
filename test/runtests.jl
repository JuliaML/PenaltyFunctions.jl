module Tests
using LearnBase, PenaltyFunctions, Base.Test
P = PenaltyFunctions

#---------------------------------------------------------------------------# Begin Tests

@testset "Common" begin
    @test P.soft_thresh(1.0, 0.5) == 0.5
    @test P.soft_thresh!(ones(5), .5) == .5 * ones(5)
    @test P.name(L1Penalty()) == "L1Penalty"
end


@testset "ElementPenalty" begin
    function test_element_penalty(p::Penalty, θ::Number, s::Number, v1, v2, v3)
        @testset "$(P.name(p))" begin
            @test value(p, θ)   ≈ v1
            @test deriv(p, θ)   ≈ v2
            @test prox(p, θ, s) ≈ v3
        end
    end

    θ, s = rand(), rand()

    test_element_penalty(NoPenalty(), θ, s, 0.0, 0.0, θ)
    test_element_penalty(L1Penalty(), θ, s, abs(θ), sign(θ), max(0.0, θ - sign(θ) * s))
    test_element_penalty(L2Penalty(), θ, s, .5 * θ^2, θ, θ / (1 + s))
    test_element_penalty(ElasticNetPenalty(.4), θ, s,
        .4 * value(L1Penalty(), θ) + .6 * value(L2Penalty(), θ),
        .4 * deriv(L1Penalty(), θ) + .6 * deriv(L2Penalty(), θ),
        prox(L2Penalty(), prox(L1Penalty(), θ, .4s), .6s)
    )

    @testset "SCADPenalty" begin
        p = SCADPenalty(3.8)

        @test value(p, .1, .2) ≈ .02
        @test value(p, .2, .1) ≈ -.5 * (.2^2 - .2^2 * 3.8 + .01) / (2.8)
        @test value(p, 9., .1) ≈ .5 * 4.8 * .01

        @test deriv(p, .1, .2) ≈ .2
        @test deriv(p, .2, .1) ≈ .1 * (3.8 * .1 - .2) / (2.8 * .1)
        @test deriv(p, 9., .1) ≈ 0.

        @test prox(p, .1, .2) ≈ 0.
        @test prox(p, .2, .1) ≈ (2.8 * .2 - 3.8 * .1) / 1.8
        @test prox(p, 9., .1) ≈ 9.
    end

    @testset "ElementPenalty methods" begin
        p = L1Penalty()
        θ, s = rand(10), rand(10)
        @testset "value" begin
            @test value(p, θ)       ≈ sum(abs, θ)
            @test value(p, θ, s[1]) ≈ s[1] * sum(abs, θ)
            @test value(p, θ, s)    ≈ sum(s .* abs.(θ))
        end
        @testset "deriv/grad" begin
            @test deriv(p, θ[1], s[1])  ≈ s[1] * sign(θ[1])
            @test grad(p, θ)            ≈ sign.(θ)
            @test grad(p, θ, s[1])      ≈ s[1] * sign.(θ)
            @test grad(p, θ, s)         ≈ s .* sign.(θ)

            buffer = rand(10)
            grad!(buffer, p, θ); @test buffer       ≈ sign.(θ)
            grad!(buffer, p, θ, s[1]); @test buffer ≈ s[1] * sign.(θ)
            grad!(buffer, p, θ, s); @test buffer    ≈ s .* sign.(θ)

            addgrad(buffer[1], p, θ[1]) ≈ buffer[1] + sign(θ[1])
            addgrad(buffer[1], p, θ[1], s[1]) ≈ buffer[1] + s[1] * sign(θ[1])
            ∇ = rand(10)
            ∇2 = copy(∇)
            addgrad!(∇, p, θ); @test ∇ ≈ ∇2 + sign.(θ)
            ∇ = rand(10)
            ∇2 = copy(∇)
            addgrad!(∇, p, θ, s[1]); @test ∇ ≈ ∇2 + s[1] * sign.(θ)
            ∇ = rand(10)
            ∇2 = copy(∇)
            addgrad!(∇, p, θ, s); @test ∇ ≈ ∇2 + s .* sign.(θ)
        end
        @testset "prox" begin
            p = L1Penalty()
            θ2 = copy(θ)
            prox!(p, θ, s[1]); @test θ ≈ P.soft_thresh.(θ2, s[1])
            θ = copy(θ2)
            prox!(p, θ, s); @test θ ≈ P.soft_thresh.(θ2, s)
        end
    end
end

end
