"""
Penalties that are applied element-wise.
"""
abstract type ElementPenalty <: Penalty end
abstract type ConvexElementPenalty <: ElementPenalty end  # only these have prox method

#-------------------------------------------------------------------------------# methods
value(p::ElementPenalty, θ::Number, s::Number)       = s * value(p, θ)
value(p::ElementPenalty, θ::AA{<:Number})            = sum(x -> value(p, x), θ)
value(p::ElementPenalty, θ::AA{<:Number}, s::Number) = sum(x -> value(p, x, s), θ)
function value{T <: Number,S <: Number}(p::ElementPenalty, θ::AA{T}, s::AA{S})
    @assert size(θ) == size(s)
    # TODO: make this work: Core.Inference.return_type(value, (typeof(p), T, S))
    result = zero(value(p, first(θ), first(s)))
    @inbounds for i in eachindex(θ, s)
        result += value(p, θ[i], s[i])
    end
    result
end

prox!(p::ConvexElementPenalty, θ::AA{<:Number}, s::Number) = map!(θj -> prox(p, θj, s), θ, θ)
function prox!(p::ConvexElementPenalty, θ::AA{<:Number}, s::AA{<:Number})
    @assert size(θ) == size(s)
    @inbounds for i in eachindex(θ, s)
        θ[i] = prox(p, θ[i], s[i])
    end
    θ
end
prox(p::ConvexElementPenalty, θ::AA{<:Number}, s::Number)       = prox!(p, copy(θ), s)
prox(p::ConvexElementPenalty, θ::AA{<:Number}, s::AA{<:Number}) = prox!(p, copy(θ), s)

deriv(p::ElementPenalty, θ::Number, s::Number) = s * deriv(p, θ)
grad(p::ElementPenalty, θ::AA{<:Number})       = grad!(similar(θ), p, θ)
grad{T<:Number,S<:Number}(p::ElementPenalty, θ::AA{T}, s::S)     = grad!(similar(θ, float(promote_type(T, S))), p, θ, s)
grad{T<:Number,S<:Number}(p::ElementPenalty, θ::AA{T}, s::AA{S}) = grad!(similar(θ, float(promote_type(T, S))), p, θ, s)
grad!(storage::AA{<:Number}, p::ElementPenalty, θ::AA{<:Number}) = map!(x -> deriv(p, x), storage, θ)
function grad!(storage::AA{<:Number}, p::ElementPenalty, θ::AA{<:Number}, s::Number)
    map!(x -> deriv(p, x, s), storage, θ)
end
function grad!(storage::AA{<:Number}, p::ElementPenalty, θ::AA{<:Number}, s::AA{<:Number})
    @assert size(storage) == size(θ) == size(s)
    @inbounds for j in eachindex(θ, s)
        storage[j] = deriv(p, θ[j], s[j])
    end
    storage
end

addgrad(∇j::Number, p::ElementPenalty, θj::Number) = ∇j + deriv(p, θj)
addgrad(∇j::Number, p::ElementPenalty, θj::Number, s::Number) = ∇j + s * deriv(p, θj)
function addgrad!(∇::AA{<:Number}, p::ElementPenalty, θ::AA{<:Number})
    @assert size(∇) == size(θ)
    @inbounds for j in eachindex(∇, θ)
        ∇[j] = addgrad(∇[j], p, θ[j])
    end
    ∇
end
function addgrad!(∇::AA{<:Number}, p::ElementPenalty, θ::AA{<:Number}, s::Number)
    @assert size(∇) == size(θ)
    @inbounds for j in eachindex(∇, θ)
        ∇[j] = addgrad(∇[j], p, θ[j], s)
    end
    ∇
end
function addgrad!(∇::AA{<:Number}, p::ElementPenalty, θ::AA{<:Number}, s::AA{<:Number})
    @assert size(∇) == size(θ) == size(s)
    @inbounds for j in eachindex(∇, θ, s)
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
prox(p::NoPenalty,  θ::Number, s::Number) = θ


"""
L1Penalty aka LASSO

`g(θ) = abs(θ)`
"""
immutable L1Penalty <: ConvexElementPenalty end
value(p::L1Penalty, θ::Number) = abs(θ)
deriv(p::L1Penalty, θ::Number) = sign(θ)
prox(p::L1Penalty,  θ::Number, s::Number) = soft_thresh(θ, s)


"""
L2Penalty aka Ridge

`g(θ) = .5 * θ ^ 2`
"""
immutable L2Penalty <: ConvexElementPenalty end
value{T <: Number}(p::L2Penalty, θ::T) = (T(1)/T(2)) * θ * θ
deriv(p::L2Penalty, θ::Number) = θ
prox{T <: Number}(p::L2Penalty, θ::T, s::Number) = θ / (one(T) + s)


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
for f in (:value, :deriv)
    @eval function ($f){T <: Number}(p::ElasticNetPenalty{T}, θ::Number)
        p.α * ($f)(L1Penalty(), θ) + (one(T) - p.α) * ($f)(L2Penalty(), θ)
    end
end
function prox{T <: Number}(p::ElasticNetPenalty{T}, θ::Number, s::Number)
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
value(p::LogPenalty, θ::Number) = log1p(p.η * abs(θ))
deriv{T <: Number}(p::LogPenalty{T}, θ::Number) = p.η * sign(θ) / (one(T) + p.η * abs(θ))



# http://www.pstat.ucsb.edu/student%20seminar%20doc/SCAD%20Jian%20Shi.pdf
"""
Smoothly Clipped Absolute Deviation Penalty
"""
immutable SCADPenalty{T <: Number} <: ElementPenalty
    a::T
    γ::T
    function SCADPenalty{T}(a::T, γ::T) where T<:Number
        a > 2 || throw(ArgumentError("First parameter must be > 2"))
        γ > 0 || throw(ArgumentError("Second parameter must be > 0"))
        new{T}(a, γ)
    end
end
SCADPenalty{T<:Number}(a::T = 3.7, γ::T = T(1)) = SCADPenalty{T}(a, γ)
SCADPenalty(a::Number, γ::Number)  = SCADPenalty(promote(a, γ)...)

function value{T,S<:Number}(p::SCADPenalty{T}, θ::S)
    absθ = abs(θ)
    R = float(promote_type(T, S))
    if absθ < p.γ
        R(p.γ * absθ)
    elseif absθ <= p.γ * p.a
        -R(0.5) * (absθ^2 - R(2) * p.a * p.γ * absθ + p.γ^2) / (p.a - one(R))
    else
        R(0.5) * (p.a + one(R)) * p.γ * p.γ
    end
end
function deriv{T,S<:Number}(p::SCADPenalty{T}, θ::S)
    absθ = abs(θ)
    R = float(promote_type(T, S))
    if absθ < p.γ
        R(p.γ * sign(θ))
    elseif absθ <= p.γ * p.a
        R(-(absθ - p.a * p.γ * sign(θ)) / (p.a - one(R)))
    else
        zero(R)
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
function value{T <: Number}(p::MCPPenalty{T}, θ::Number)
    t = abs(θ)
    t < p.γ ? t - t^2 / (T(2) * p.γ) : (T(1)/T(2)) * p.γ
end
function deriv{T <: Number, S <: Number}(p::MCPPenalty{T}, θ::S)
    t = abs(θ)
    t < p.γ ? sign(θ) * (T(1) - t / p.γ): zero(float(promote_type(S,T)))
end













#--------------------------------------------------------------------------------# scaled
function _scale_check(λ)
    isa(λ, Number) || throw(ArgumentError("Scale factor λ must be a Number"))
    λ >= 0 || throw(ArgumentError("Scale factor λ has to be strictly positive."))
end

immutable ScaledElementPenalty{T <: Number, P <: ElementPenalty} <: ElementPenalty
    penalty::P
    λ::T
end
scaled(p::ElementPenalty, λ::Number) = (_scale_check(λ); ScaledElementPenalty(p, λ))
Base.show(io::IO, sp::ScaledElementPenalty) = print(io, "$(sp.λ) * ($(sp.penalty))")


value(p::ScaledElementPenalty{<:Number}, θ::Number) = p.λ * value(p.penalty, θ)
deriv(p::ScaledElementPenalty{<:Number}, θ::Number) = p.λ * deriv(p.penalty, θ)
prox(p::ScaledElementPenalty{<:Number},  θ::Number) = prox(p.penalty, θ, p.λ)
prox(p::ScaledElementPenalty{<:Number},  θ::AA{<:Number}) = prox(p.penalty, θ, p.λ)

# SCAD is special
for f in (:value, :deriv)
    @eval function ($f)(p::ScaledElementPenalty{<:Number, <:SCADPenalty}, θ::Number)
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
