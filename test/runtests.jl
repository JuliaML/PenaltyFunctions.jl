module Tests
using LearnBase, PenaltyFunctions, Base.Test


# Test value, in the order:
# value(pen, θ[1])            scalar
# value(pen, θ[1], s[1])      scalar, weighted by scalar
# value(pen, θ)               array
# value(pen, θ, s[1])         array, weighted by scalar
# value(pen, θ, s)            array, weighted by array
function test_value(pen, θ, s, v1, v2, v3, v4, v5)
    @test value(pen, θ[1])        ≈ v1
    @test value(pen, θ[1], s[1])  ≈ v2
    @test value(pen, θ)           ≈ v3
    @test value(pen, θ, s[1])     ≈ v4
    @test value(pen, θ, s)        ≈ v5
end

# test deriv/grad, in the order:
# deriv(pen, θ[1])                scalar
# deriv(pen, θ[1], s[1])          scalar, weighted by scalar
# grad(pen, θ)                    array
# grad!(buffer, pen, θ)           array, overwrite buffer
# grad!(buffer, pen, θ, s[1])     array, weighted by scalar, overwrite buffer
# grad!(buffer, pen, θ, s)        array, weighted by array, overwrite buffer
function test_deriv(pen, θ, s, buffer, v1, v2, v3, v4, v5, v6)
    @test deriv(pen, θ[1])                      ≈ v1
    @test deriv(pen, θ[1], s[1])                ≈ v2
    @test grad(pen, θ)                          ≈ v3
    grad!(buffer, pen, θ);        @test buffer  ≈ v4
    grad!(buffer, pen, θ, s[1]);  @test buffer  ≈ v5
    grad!(buffer, pen, θ, s);     @test buffer  ≈ v6
end

# test prox/prox!, in the order
# prox(pen, θ[1], s[1])     scalar, weighted by scalar
# prox!(pen, θ, s[1])       array, weighted by scalar
# prox!(pen, θ, s)          array, weighted by array
function test_prox(pen, θ, s, v1, v2, v3)
    @test prox(pen, θ[1], s[1])     ≈ v1
    prox!(pen, θ, s[1]); @test θ    ≈ v2
    prox!(pen, θ, s); @test θ       ≈ v3
end

function test_addgrad(pen, θ, s, buffer, v1, v2, v3)
    @test addgrad(.1, pen, θ[1])              ≈ v1
    @test addgrad(.1, pen, θ[1], .1)          ≈ v2
    addgrad!(buffer, pen, θ); @test buffer    ≈ v3
end

#---------------------------------------------------------------------------# Begin Tests

@testset "Common" begin
    @test PenaltyFunctions.soft_thresh(1.0, 0.5) == 0.5
    @test PenaltyFunctions.soft_thresh!(ones(5), .5) == .5 * ones(5)
    @test PenaltyFunctions.name(L1Penalty()) == "L1Penalty"
end

@testset "ElementPenalty" begin
    @testset "NoPenalty" begin
        p = NoPenalty()
        θ, s, buffer = rand(10), rand(10), rand(10)

        test_value(p, θ, s, zeros(5)...)
        test_deriv(p, θ, s, buffer, 0.0, 0.0, zeros(10), zeros(10), zeros(10), zeros(10))
        test_prox(p, θ, s, θ[1], copy(θ), copy(θ))
        test_addgrad(p, θ, s, buffer, .1, .1, copy(buffer))
    end
    @testset "L1Penalty" begin
        p = L1Penalty()
        θ = rand(10)
        θ2 = copy(θ)
        s = rand(10)
        buffer = rand(10)

        test_value(p, θ, s, abs(θ[1]), s[1] * abs(θ[1]), sum(abs, θ), s[1] * sum(abs, θ),
            sum(s .* abs.(θ)))
        test_deriv(p, θ, s, buffer, sign(θ[1]), s[1] * sign(θ[1]), sign(θ), sign.(θ),
            s[1] * sign.(θ), s .* sign.(θ))

    end
end

end
