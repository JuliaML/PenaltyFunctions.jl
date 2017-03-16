"""
Penalties that are applied element-wise.
"""
abstract type ElementPenalty <: Penalty end
abstract type ConvexElementPenalty <: ElementPenalty end  # only these have prox method

# Make broadcast work for ElementPenalty
Base.getindex(p::ElementPenalty, idx) = p
Base.size(::ElementPenalty) = ()

#-------------------------------------------------------------------------------# methods
value{T}(p::ElementPenalty, θ::T, s::T)     = s * value(p, θ)
value{T}(p::ElementPenalty, θ::AA{T})       = sum(x -> value(p, x), θ)
value{T}(p::ElementPenalty, θ::AA{T}, s::T) = sum(x -> value(p, x, s), θ)
function value{T}(p::ElementPenalty, θ::AA{T}, s::AA{T})
    @assert size(θ) == size(s)
    result = zero(T)
    for i in eachindex(θ)
        @inbounds result += value(p, θ[i], s[i])
    end
    result
end

prox!{T}(p::ConvexElementPenalty, θ::AA{T}, s::T) = map!(θj -> prox(p, θj, s), θ, θ)
function prox!{T}(p::ConvexElementPenalty, θ::AA{T}, s::AA{T})
    @assert size(θ) == size(s)
    for i in eachindex(θ)
        @inbounds θ[i] = prox(p, θ[i], s[i])
    end
    θ
end
prox{T}(p::ConvexElementPenalty, θ::AA{T}, s::T)      = prox!(p, copy(θ), s)
prox{T}(p::ConvexElementPenalty, θ::AA{T}, s::AA{T})  = prox!(p, copy(θ), s)

deriv{T}(p::ElementPenalty, θ::T, s::T) = s * deriv(p, θ)
grad{T}(p::ElementPenalty, θ::AA{T})                = grad!(similar(θ), p, θ)
grad{T}(p::ElementPenalty, θ::AA{T}, s::T)          = grad!(similar(θ), p, θ, s)
grad{T}(p::ElementPenalty, θ::AA{T}, s::AA{T})      = grad!(similar(θ), p, θ, s)
grad!{T}(storage::AA{T}, p::ElementPenalty, θ::AA{T}) = map!(x -> deriv(p, x), storage, θ)
function grad!{T}(storage::AA{T}, p::ElementPenalty, θ::AA{T}, s::T)
    map!(x -> deriv(p, x, s), storage, θ)
end
function grad!{T}(storage::AA{T}, p::ElementPenalty, θ::AA{T}, s::AA{T})
    @assert size(storage) == size(θ) == size(s)
    for j in eachindex(θ)
        @inbounds storage[j] = deriv(p, θ[j], s[j])
    end
    storage
end

addgrad{T}(∇j::T, p::ElementPenalty, θj::T) = ∇j + deriv(p, θj)
addgrad{T}(∇j::T, p::ElementPenalty, θj::T, s::T) = ∇j + s * deriv(p, θj)
function addgrad!{T}(∇::AA{T}, p::ElementPenalty, θ::AA{T})
    @assert size(∇) == size(θ)
    @inbounds for j in eachindex(∇)
        ∇[j] = addgrad(∇[j], p, θ[j])
    end
    ∇
end
function addgrad!{T}(∇::AA{T}, p::ElementPenalty, θ::AA{T}, s::T)
    @assert size(∇) == size(θ)
    @inbounds for j in eachindex(∇)
        ∇[j] = addgrad(∇[j], p, θ[j], s)
    end
    ∇
end
function addgrad!{T}(∇::AA{T}, p::ElementPenalty, θ::AA{T}, s::AA{T})
    @assert size(∇) == size(θ) == size(s)
    @inbounds for j in eachindex(∇)
        ∇[j] = addgrad(∇[j], p, θ[j], s[j])
    end
    ∇
end

#----------------------------------------------------------------------# ElementPenalties
"""
Unpenalized

`g(θ) = 0`
"""
immutable NoPenalty <: ConvexElementPenalty end
value(p::NoPenalty, θ::Number) = zero(θ)
deriv(p::NoPenalty, θ::Number) = zero(θ)
prox{T <: Number}(p::NoPenalty, θ::T, s::T) = θ


"""
L1Penalty aka LASSO

`g(θ) = abs(θ)`
"""
immutable L1Penalty <: ConvexElementPenalty end
value(p::L1Penalty, θ::Number) = abs(θ)
deriv(p::L1Penalty, θ::Number) = sign(θ)
prox{T <: Number}(p::L1Penalty, θ::T, s::T) = soft_thresh(θ, s)


"""
L2Penalty aka Ridge

`g(θ) = .5 * θ ^ 2`
"""
immutable L2Penalty <: ConvexElementPenalty end
value(p::L2Penalty, θ::Number) = typeof(θ)(0.5) * θ * θ
deriv(p::L2Penalty, θ::Number) = θ
prox{T <: Number}(p::L2Penalty, θ::T, s::T) = θ / (one(T) + s)


"""
ElasticNetPenalty, weighted average of L1Penalty and L2Penalty

`g(θ) = α * abs(θ) + (1 - α) * .5 * θ ^ 2`
"""
immutable ElasticNetPenalty{T <: Number} <: ConvexElementPenalty
    α::T
    function ElasticNetPenalty(α::T = 0.5) where T <: Number
        0 <= α <= 1 || throw(ArgumentError("α must be in [0, 1]"))
        new{T}(α)
    end
end
for f in [:value, :deriv]
    @eval function ($f){T <: Number}(p::ElasticNetPenalty{T}, θ::T)
        p.α * ($f)(L1Penalty(), θ) + (1 - p.α) * ($f)(L2Penalty(), θ)
    end
end
function prox{T <: Number}(p::ElasticNetPenalty{T}, θ::T, s::T)
    αs = p.α * s
    soft_thresh(θ, αs) / (one(T) + s - αs)
end


"""
LogPenalty(η)

`g(θ) = log(1 + η * θ)`
"""
immutable LogPenalty{T <: Number} <: ElementPenalty
    η::T
    function LogPenalty(η::T = 1.0) where T <: Number
        η > 0 || throw(ArgumentError("η must be > 0"))
        new{T}(η)
    end
end
value{T}(p::LogPenalty{T}, θ::T) = log(1 + p.η * abs(θ))
deriv{T}(p::LogPenalty{T}, θ::T) = p.η * sign(θ) / (1 + p.η * abs(θ))



# http://www.pstat.ucsb.edu/student%20seminar%20doc/SCAD%20Jian%20Shi.pdf
"""
Smoothly Clipped Absolute Deviation Penalty
"""
immutable SCADPenalty{T <: Number} <: ElementPenalty
    a::T
    γ::T
    function SCADPenalty(a::T = 3.7, γ::T = 1.0) where T
        a > 2 || throw(ArgumentError("First parameter must be > 2"))
        γ > 0 || throw(ArgumentError("Second parameter must be > 0"))
        new{T}(a, γ)
    end
end
function value{T}(p::SCADPenalty{T}, θ::T)
    absθ = abs(θ)
    if absθ < p.γ
        return p.γ * absθ
    elseif absθ <= p.γ * p.a
        return -T(0.5) * (absθ ^ 2 - 2 * p.a * p.γ * absθ + p.γ ^ 2) / (p.a - one(T))
    else
        return T(0.5) * (p.a + 1) * p.γ * p.γ
    end
end
function deriv{T}(p::SCADPenalty{T}, θ::T)
    absθ = abs(θ)
    if absθ < p.γ
        return p.γ * sign(θ)
    elseif absθ <= p.γ * p.a
        return -(absθ - p.a * p.γ * sign(θ)) / (p.a - 1)
    else
        return zero(T)
    end
end
# function prox{T}(p::SCADPenalty{T}, θ::T, λ::T)
#     absθ = abs(θ)
#     if absθ < 2λ
#         return prox(L1Penalty(), θ, λ)
#     elseif absθ <= λ * p.a
#         return prox(L1Penalty(), θ, λ * p.a / (p.a - 1)) * (p.a - 1) / (p.a - 2)
#     else
#         return θ
#     end
# end


# https://arxiv.org/abs/1002.4734
"""
MCPPenalty(γ) (MC+)
"""
immutable MCPPenalty{T <: Number} <: ElementPenalty
    γ::T  # In paper, this is λ * γ
    function MCPPenalty(γ::T = 2.0) where T
        γ > 0 || throw(ArgumentError("γ must be > 0"))
        new{T}(γ)
    end
end
MCPPenalty(γ::Integer) = MCPPenalty(Float64(γ))
function value{T}(p::MCPPenalty{T}, θ::T)
    t = abs(θ)
    t < p.γ ? t - t^2 / (2 * p.γ) : T(0.5) * p.γ
end
function deriv{T}(p::MCPPenalty{T}, θ::T)
    t = abs(θ)
    t < p.γ ? sign(θ) * (1 - t / p.γ): 0.0
end













#--------------------------------------------------------------------------------# scaled
function _scale_check(λ)
    isa(λ, Number) || throw(ArgumentError("Scale factor λ must be a Number"))
    λ >= 0 || throw(ArgumentError("Scale factor λ has to be strictly positive."))
end

immutable ScaledElementPenalty{T, P <: ElementPenalty} <: ElementPenalty
    penalty::P
    λ::T
end
scaled(p::ElementPenalty, λ::Number) = (_scale_check(λ); ScaledElementPenalty(p, λ))
Base.show(io::IO, sp::ScaledElementPenalty) = print(io, "$(sp.λ) * ($(sp.penalty))")


value{T}(p::ScaledElementPenalty{T}, θ::T) = p.λ * value(p.penalty, θ)
deriv{T}(p::ScaledElementPenalty{T}, θ::T) = p.λ * deriv(p.penalty, θ)
prox{T}(p::ScaledElementPenalty{T}, θ::T) = prox(p.penalty, θ, p.λ)
prox{T}(p::ScaledElementPenalty{T}, θ::AA{T}) = prox(p.penalty, θ, p.λ)

# SCAD is special
for f in [:value, :deriv]
    @eval function ($f){P <: SCADPenalty, T}(p::ScaledElementPenalty{T, P}, θ::T)
        ($f)(p.penalty, θ, p.λ)
    end
end

#--------------------------------------------------------------------------------# plot
@recipe function f(p::ElementPenalty)
    label --> name(p)
    xlims --> (-4, 4)
    ylabel --> "penalty value"
    xlabel --> "parameter"
    g(x) = value(p, x, 1.)
    g
end
