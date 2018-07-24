module Tests
using PenaltyFunctions, LinearAlgebra, Test
P = PenaltyFunctions


element_penalties = [NoPenalty(), L1Penalty(), L2Penalty(),
                     ElasticNetPenalty(.7), ElasticNetPenalty(.7f0),
                     MCPPenalty(1.), MCPPenalty(1f0), MCPPenalty(1),
                     LogPenalty(1.), LogPenalty(1f0),
                     SCADPenalty(3.7), SCADPenalty(3.7f0)]
array_penalties = [NuclearNormPenalty(), GroupLassoPenalty(),
                   MahalanobisPenalty(rand(3,2))]

penalty_list = vcat(element_penalties, array_penalties)

@info "Show methods:"
for p in penalty_list
    printstyled("  > $(string(p))\n";             color = :green)
    printstyled("  > $(string(scaled(p, .1)))\n"; color = :blue)
end

#---------------------------------------------------------------------------# Begin Tests
println("\n")
@info "Begin Actual Tests"
@testset "Common" begin
    @test P.soft_thresh(1.0, 0.5) == 0.5
    @test P.soft_thresh!(ones(5), .5) == .5 * ones(5)
    @test P.name(L1Penalty()) == "L1Penalty"
end


@testset "ElementPenalty" begin
    function test_element_penalty(p::Penalty, θ::Number, s::Number, v1, v2, v3)
        r(x) = Float32(round(x; digits=5)) # avoids floating point issues for different parameter types
        @testset "$(P.name(p))" begin
            @test r(@inferred(value(p, θ))) ≈ r(v1)
            @test r(@inferred(deriv(p, θ))) ≈ r(v2)
            @test r.(value.(Ref(p), fill(θ, 5))) ≈ r.(fill(v1, 5))
            @test r.(deriv.(Ref(p), fill(θ, 5))) ≈ r.(fill(v2, 5))
            if isa(p, P.ProxableElementPenalty)
                @test r(@inferred(prox(p, θ, s))) ≈ r(v3)
                @test r.(prox.(Ref(p), fill(θ, 5), Ref(s))) ≈ r.(fill(v3, 5))
            end
        end
    end

    θ, s = rand(), rand()

    @testset "Constructors" begin
        @inferred NoPenalty()
        @inferred L1Penalty()
        @inferred L2Penalty()

        @inferred ElasticNetPenalty()
        @inferred ElasticNetPenalty(1.0)

        @inferred SCADPenalty()
        @inferred SCADPenalty(4.0)
        @inferred SCADPenalty(4)
        @inferred SCADPenalty(4, 1.)
        @inferred SCADPenalty(4.0, 1.0)
        @test_throws Exception SCADPenalty(1.0)
        @test_throws Exception SCADPenalty(3.7, -1.)
        @test typeof(SCADPenalty())         <: SCADPenalty{Float64}
        @test typeof(SCADPenalty(4))        <: SCADPenalty{Int}
        @test typeof(SCADPenalty(4, 1))     <: SCADPenalty{Int}
        @test typeof(SCADPenalty(4f0))      <: SCADPenalty{Float32}
        @test typeof(SCADPenalty(4f0, 1))   <: SCADPenalty{Float32}
        @test typeof(SCADPenalty(4f0, 1.0)) <: SCADPenalty{Float64}
        @test typeof(SCADPenalty(4, 1.0))   <: SCADPenalty{Float64}
        @test typeof(SCADPenalty(4.0, 1.0)) <: SCADPenalty{Float64}

        @inferred LogPenalty()
        @inferred LogPenalty(1.0)
        @test_throws Exception LogPenalty(-1.0)

        @inferred MCPPenalty()
        @inferred MCPPenalty(1.0)
        @test_throws Exception MCPPenalty(-1.0)
    end

    for T in (Float32, Float64), S in (Float32, Float64)
        test_element_penalty(NoPenalty(), T(θ), S(s), 0.0, 0.0, θ)
        test_element_penalty(L1Penalty(), T(θ), S(s), abs(θ), sign(θ), max(0.0, θ - sign(θ) * s))
        test_element_penalty(L2Penalty(), T(θ), S(s), .5 * θ^2, θ, θ / (1 + s))
        test_element_penalty(ElasticNetPenalty(.4), T(θ), S(s),
            .4 * value(L1Penalty(), θ) + .6 * value(L2Penalty(), θ),
            .4 * deriv(L1Penalty(), θ) + .6 * deriv(L2Penalty(), θ),
            prox(L2Penalty(), prox(L1Penalty(), θ, .4s), .6s)
        )

        test_element_penalty(LogPenalty(1.0), T(θ), S(s), log(1 + θ), 1 / (1 + θ), nothing)
        test_element_penalty(MCPPenalty(2.0), T(1.0), S(s), 1 - 1/4, 1 - 1/2, nothing)
        test_element_penalty(MCPPenalty(1.0), T(2.0), S(s), 1/2, 0.0, nothing)
    end

    @testset "SCADPenalty" begin
        for T in (Float32, Float64), S in (Float32, Float64)
            r(x) = round(x, digits=8)
            @test r(@inferred(value(SCADPenalty(T(3.8), T(.2)), S(.1)))) ≈ r(.02)
            @test r(@inferred(value(SCADPenalty(T(3.8), T(.1)), S(.2)))) ≈ r(-.5 * (.2^2 - .2^2 * 3.8 + .01) / (2.8))
            @test r(@inferred(value(SCADPenalty(T(3.8), T(.1)), S(9.)))) ≈ r(.5 * 4.8 * .01)
            @test r(@inferred(value(SCADPenalty(T(3.8), T(.1)), 9)))     ≈ r(.5 * 4.8 * .01)

            @test r(@inferred(deriv(SCADPenalty(T(3.8), T(.2)), S(.1)))) ≈ r(.2)
            @test r(@inferred(deriv(SCADPenalty(T(3.8), T(.1)), S(.2)))) ≈ r(.1 * (3.8 * .1 - .2) / (2.8 * .1))
            @test r(@inferred(deriv(SCADPenalty(T(3.8), T(.1)), S(9.)))) ≈ r(0.)
            @test r(@inferred(deriv(SCADPenalty(T(3.8), T(.1)), 9)))     ≈ r(0.)
        end
    end

    @testset "ElementPenalty methods" begin
        p = L1Penalty()
        for T in (Float32, Float64), S in (Float32, Float64)
            θ, s = rand(T, 10), rand(S, 10)
            @testset "value: $T, $S" begin
                @test round(@inferred(value(p, θ, s[1])), digits=4) ≈ round(s[1] * sum(abs, θ), digits=4)
                @test @inferred(value(p, θ))       ≈ sum(abs, θ)
                @test @inferred(value(p, θ, s))    ≈ sum(s .* abs.(θ))
            end
            @testset "deriv/grad: $T, $S" begin
                @test @inferred(deriv(p, θ[1], s[1]))  ≈ s[1] * sign(θ[1])
                @test @inferred(grad(p, θ))            ≈ sign.(θ)
                @test @inferred(grad(p, θ, s[1]))      ≈ s[1] * sign.(θ)
                @test @inferred(grad(p, θ, s))         ≈ s .* sign.(θ)
                @test deriv.(Ref(p), θ)       == grad(p, θ)
                @test deriv.(Ref(p), θ, s[1]) == grad(p, θ, s[1])
                @test deriv.(Ref(p), θ, s)    == grad(p, θ, s)

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
            @testset "prox: $T, $S" begin
                p = L1Penalty()
                θ2 = copy(θ)
                prox!(p, θ, s[1]); @test θ ≈ P.soft_thresh.(θ2, s[1])
                θ = copy(θ2)
                prox!(p, θ, s); @test θ ≈ P.soft_thresh.(θ2, s)

                θ = rand(5)
                s = rand(5)
                @test @inferred(prox(p, θ, s)) == map((x,y) -> prox(p, x, y), θ, s)
            end
        end
    end
end

@testset "ScaledElementPenalty" begin
    for p in element_penalties
        s = scaled(p, .1)
        x = randn(5)
        # FIXME: @inference broken during tests (not in REPL though)
        #        only for ElasticNetPenalty{Float64} for some reason
        @test value(s, x) ≈ value(p, x, .1)
        @test value(s, x) == value(.1 * p, x)
        @test @inferred(deriv(s, x[1])) ≈ deriv(p, x[1], .1)
        @test @inferred(grad(s, x))     ≈ grad(p, x, .1)
        if typeof(p) <: P.ProxableElementPenalty
            @test @inferred(prox(s, x)) ≈ prox(p, x, .1)
        end
    end

    p = ElasticNetPenalty(.7)
    s = scaled(p, .2)
    x = randn(5)
    # FIXME: @inference broken during tests (not in REPL though)
    @test value(s, x) ≈ .2 * value(p, x)
    @test @inferred(deriv(s, x[1])) ≈ .2 * deriv(p, x[1])
    @test @inferred(grad(s, x)) ≈ .2 * grad(p, x)
    @test deriv.(Ref(s), x) ≈ .2 * deriv.(Ref(p), x)
    @test @inferred(prox(s, x)) ≈ prox(p, x, .2)
    @test @inferred(prox(s, x[1])) ≈ prox(p, x[1], .2)

    @test_throws ArgumentError scaled(p, -1.)
end

@testset "ArrayPenalty" begin
    @testset "NuclearNormPenalty" begin
        p = NuclearNormPenalty()
        Θ = randn(10, 5)
        s = .05
        # FIXME: @inference broken. seems like a type instability
        @test value(p, Θ) ≈ sum(svd(Θ).S)
        @test value(p, Θ, s) ≈ s * sum(svd(Θ).S)
        prox!(p, Θ, s)
    end
    @testset "GroupLassoPenalty" begin
        p = GroupLassoPenalty()
        Θ = randn(10)
        s = .05
        @test @inferred(value(p, Θ)) ≈ norm(Θ)
        prox!(p, Θ, s)

        Θ = .01 * ones(10)
        prox!(p, Θ, 10.) == zeros(10)
    end
    @testset "MahalanobisPenalty" begin
        C = randn(5, 10)
        p = MahalanobisPenalty(C)
        θ = rand(10)
        s = .05
        @test @inferred(value(p, θ)) ≈ 0.5 * dot(C * θ, C * θ)
        prox!(p, θ, s)
    end
    @testset "ScaledArrayPenalty" begin
        p = GroupLassoPenalty()
        s = scaled(p, .1)
        Θ = randn(10)
        @test @inferred(value(p, Θ, .1)) ≈ value(s, Θ)

        Θ2 = copy(Θ)
        prox!(p, Θ, .1); prox!(s, Θ2); @test Θ ≈ Θ2
    end
end

end  # module
