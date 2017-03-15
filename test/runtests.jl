module Tests
using PenaltyFunctions, Base.Test
P = PenaltyFunctions


element_penalties = [NoPenalty(), L1Penalty(), L2Penalty(),
                     ElasticNetPenalty(.7), MCPPenalty(1.),
                     LogPenalty(1.), SCADPenalty(3.7)]
array_penalties = [NuclearNormPenalty(), GroupLassoPenalty(),
                   MahalanobisPenalty(rand(3,2))]

penalty_list = vcat(element_penalties, array_penalties)

info("Show methods:")
for p in penalty_list
    print_with_color(:red, "  > $(string(p))\n")
    print_with_color(:cyan, "  > $(string(scaled(p, .1)))\n")
end

#---------------------------------------------------------------------------# Begin Tests
println("\n")
info("Begin Actual Tests")
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
            @test value.(p, fill(θ, 5)) ≈ fill(v1, 5)
            @test deriv.(p, fill(θ, 5)) ≈ fill(v2, 5)
            if isa(p, ConvexElementPenalty)
                @test prox(p, θ, s) ≈ v3
                @test prox.(p, fill(θ, 5), s) ≈ fill(v3, 5)
            end
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

    test_element_penalty(LogPenalty(1.0), θ, s, log(1 + θ), 1 / (1 + θ), nothing)
    test_element_penalty(MCPPenalty(2.0), 1.0, s, 1 - 1/4, 1 - 1/2, nothing)
    test_element_penalty(MCPPenalty(1.0), 2.0, s, 1/2, 0.0, nothing)

    @testset "SCADPenalty" begin
        @test value(SCADPenalty(3.8, .2), .1) ≈ .02
        @test value(SCADPenalty(3.8, .1), .2) ≈ -.5 * (.2^2 - .2^2 * 3.8 + .01) / (2.8)
        @test value(SCADPenalty(3.8, .1), 9.) ≈ .5 * 4.8 * .01

        @test deriv(SCADPenalty(3.8, .2), .1) ≈ .2
        @test deriv(SCADPenalty(3.8, .1), .2) ≈ .1 * (3.8 * .1 - .2) / (2.8 * .1)
        @test deriv(SCADPenalty(3.8, .1), 9.) ≈ 0.
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
            @test deriv.(p, θ)       == grad(p, θ)
            @test deriv.(p, θ, s[1]) == grad(p, θ, s[1])
            @test deriv.(p, θ, s)    == grad(p, θ, s)

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

            θ = rand(5)
            s = rand(5)
            @test prox(p, θ, s) == map((x,y) -> prox(p, x, y), θ, s)
        end
    end
end

@testset "ScaledElementPenalty" begin
    for p in element_penalties
        s = scaled(p, .1)
        x = randn(5)
        @test value(s, x)       ≈ value(p, x, .1)
        @test deriv(s, x[1])    ≈ deriv(p, x[1], .1)
        @test grad(s, x)        ≈ grad(p, x, .1)
        if typeof(p) <: ConvexElementPenalty
            @test prox(s, x)        ≈ prox(p, x, .1)
        end
    end

    p = ElasticNetPenalty(.7)
    s = scaled(p, .2)
    x = randn(5)
    @test value(s, x) ≈ .2 * value(p, x)
    @test deriv(s, x[1]) ≈ .2 * deriv(p, x[1])
    @test grad(s, x) ≈ .2 * grad(p, x)
    @test deriv.(s, x) ≈ .2 * deriv.(p, x)
    @test prox(s, x) ≈ prox(p, x, .2)
    @test prox(s, x[1]) ≈ prox(p, x[1], .2)

    @test_throws ArgumentError scaled(p, -1.)
end

@testset "ArrayPenalty" begin
    @testset "NuclearNormPenalty" begin
        p = NuclearNormPenalty()
        Θ = randn(10, 5)
        s = .05
        @test value(p, Θ) ≈ sum(svd(Θ)[2])
        @test value(p, Θ, s) ≈ s * sum(svd(Θ)[2])
        prox!(p, Θ, s)
    end
    @testset "GroupLassoPenalty" begin
        p = GroupLassoPenalty()
        Θ = randn(10)
        s = .05
        @test value(p, Θ) ≈ vecnorm(Θ)
        prox!(p, Θ, s)

        Θ = .01 * ones(10)
        prox!(p, Θ, 10.) == zeros(10)
    end
    @testset "MahalanobisPenalty" begin
        C = randn(5, 10)
        p = MahalanobisPenalty(C)
        θ = rand(10)
        s = .05
        @test value(p, θ) ≈ 0.5 * dot(C * θ, C * θ)
        prox!(p, θ, s)
    end
    @testset "ScaledArrayPenalty" begin
        p = GroupLassoPenalty()
        s = scaled(p, .1)
        Θ = randn(10)
        @test value(p, Θ, .1) ≈ value(s, Θ)

        Θ2 = copy(Θ)
        prox!(p, Θ, .1); prox!(s, Θ2); @test Θ ≈ Θ2
    end
end

end  # module
