module Tests
using LearnBase, PenaltyFunctions, Base.Test

@testset "ElementPenalty" begin
    @testset "NoPenalty" begin
        p = NoPenalty()
        θ = rand(10)
        θ2 = copy(θ)
        s = rand(10)
        buffer = rand(10)

        @test value(p, θ[1])        == 0.0
        @test value(p, θ[1], s[1])  == 0.0
        @test value(p, θ)           == 0.0
        @test value(p, θ, s[1])     == 0.0
        @test value(p, θ, s)        == 0.0

        @test deriv(p, θ[1])        == 0.0
        @test deriv(p, θ[1], s[1])  == 0.0
        @test grad(p, θ)            == zeros(10)
        grad!(buffer, p, θ);        @test buffer == zeros(10)
        grad!(buffer, p, θ, s[1]);  @test buffer == zeros(10)
        grad!(buffer, p, θ, s);     @test buffer == zeros(10)

        @test prox(p, θ[1], s[1]) == θ[1]
        prox!(p, θ, s[1]); @test θ == θ2
        prox!(p, θ, s); @test θ == θ2

        @test addgrad(.1, p, θ[1]) == .1
        @test addgrad(.1, p, θ[1], .1) == .1
        buffer2 = copy(buffer)
        addgrad!(buffer, p, θ); @test buffer == buffer2
    end
    @testset "L1Penalty" begin
        p = L1Penalty()
        θ = rand(10)
        θ2 = copy(θ)
        s = rand(10)
        buffer = rand(10)

        @test value(p, θ[1])        ≈ abs(θ[1])
        @test value(p, θ[1], s[1])  ≈ s[1] * abs(θ[1])
        @test value(p, θ)           ≈ sum(abs, θ)
        @test value(p, θ, s[1])     ≈ s[1] * sum(abs, θ)
        @test value(p, θ, s)        ≈ sum(s .* abs.(θ))
    end
end

end
