"""
Penalties that are applied element-wise.
"""
abstract ElementPenalty <: Penalty

#-------------------------------------------------------------------------------# methods
value{T}(p::ElementPenalty, θ::T, s::T)     = s * value(p, θ)
value{T}(p::ElementPenalty, θ::AA{T})       = sum(x -> value(p, x), θ)
value{T}(p::ElementPenalty, θ::AA{T}, s::T) = sum(x -> value(p, x, s), θ)
function value{T}(p::ElementPenalty, θ::AA{T}, s::AA{T})
    @assert size(θ) == size(s)
    sum(map((x,y) -> value(p, x, y), θ, s))
end

prox!{T}(p::ElementPenalty, θ::AA{T}, s::T) = map!(θj -> prox(p, θj, s), θ)
function prox!{T}(p::ElementPenalty, θ::AA{T}, s::AA{T})
    @assert size(θ) == size(s)
    for i in eachindex(θ)
        @inbounds θ[i] = prox(p, θ[i], s[i])
    end
    θ
end

deriv{T}(p::ElementPenalty, θ::T, s::T) = s * deriv(p, θ)
grad{T}(p::ElementPenalty, θ::AA{T})                = grad!(similar(θ), p, θ)
grad{T}(p::ElementPenalty, θ::AA{T}, s::T)          = grad!(similar(θ), p, θ, s)
grad{T}(p::ElementPenalty, θ::AA{T}, s::AA{T})      = grad!(similar(θ), p, θ, s)
grad!{T}(storage::AA{T}, p::ElementPenalty, θ::AA{T}) = map!(x -> deriv(p, x), storage, θ)
function grad!{T}(storage::AA{T}, p::ElementPenalty, θ::AA{T}, s::T)
    grad!(storage, p, θ)
    scale!(storage, s)
end
function grad!{T}(storage::AA{T}, p::ElementPenalty, θ::AA{T}, s::AA{T})
    @assert size(θ) == size(s)
    grad!(storage, p, θ)
    storage .*= s
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
"""
immutable NoPenalty <: ElementPenalty end
value(p::NoPenalty, θ::Number) = zero(θ)
deriv(p::NoPenalty, θ::Number) = zero(θ)
prox{T <: Number}(p::NoPenalty, θ::T, s::T) = θ

"""
L1Penalty aka LASSO
"""
immutable L1Penalty <: ElementPenalty end
value(p::L1Penalty, θ::Number) = abs(θ)
deriv(p::L1Penalty, θ::Number) = sign(θ)
prox{T <: Number}(p::L1Penalty, θ::T, s::T) = soft_thresh(θ, s)

"""
L2Penalty aka Ridge
"""
immutable L2Penalty <: ElementPenalty end
value(p::L2Penalty, θ::Number) = typeof(θ)(0.5) * θ * θ
deriv(p::L2Penalty, θ::Number) = θ
prox{T <: Number}(p::L2Penalty, θ::T, s::T) = θ / (one(T) + s)

"""
ElasticNetPenalty, weighted average of L1Penalty and L2Penalty
"""
immutable ElasticNetPenalty{T <: Number} <: ElementPenalty α::T end
ElasticNetPenalty(α::Number) = (@assert 0 <= α <= 1; ElasticNetPenalty(α))
for f in [:value, :deriv]
    @eval function ($f){T <: Number}(p::ElasticNetPenalty{T}, θ::T)
        p.α * ($f)(L1Penalty(), θ) + (1 - p.α) * ($f)(L2Penalty(), θ)
    end
end
function prox{T <: Number}(p::ElasticNetPenalty{T}, θ::T, s::T)
    αs = p.α * s
    soft_thresh(θ, αs) / (one(T) + s - αs)
end

#-----------------------------------------------------------------------# SCADPenalty
# http://www.pstat.ucsb.edu/student%20seminar%20doc/SCAD%20Jian%20Shi.pdf
"""
Smoothly Clipped Absolute Deviation Penalty
"""
immutable SCADPenalty{T <: Number} <: ElementPenalty
    a::T
end
SCADPenalty(a::Number = 3.7) = (@assert a > 2; SCADPenalty(a))
function value{T <: Number}(p::SCADPenalty{T}, θ::T, λ::T)
    absθ = abs(θ)
    if absθ < λ
        return λ * absθ
    elseif absθ <= λ * p.a
        return -T(0.5) * (absθ ^ 2 - 2 * p.a * λ * absθ + λ ^ 2) / (p.a - one(T))
    else
        return T(0.5) * (p.a + 1) * λ * λ
    end
end
function deriv{T <: Number}(p::SCADPenalty{T}, θ::T, λ::T)
    absθ = abs(θ)
    if absθ < λ
        return λ * sign(θ)
    elseif absθ <= λ * p.a
        return -(absθ - p.a * λ * sign(θ)) / (p.a - 1)
    else
        return zero(T)
    end
end
function prox{T <: Number}(p::SCADPenalty{T}, θ::T, λ::T)
    absθ = abs(θ)
    if absθ < 2λ
        return prox(L1Penalty(), θ, λ)
    elseif absθ <= λ * p.a
        return prox(L1Penalty(), θ, λ * p.a / (p.a - 1)) * (p.a - 1) / (p.a - 2)
    else
        return θ
    end
end



#--------------------------------------------------------------------------------# scaled
_scaled_error() = throw(ArgumentError("Scale factor λ has to be strictly positive."))
immutable ScaledElementPenalty{P <: ElementPenalty, λ} <: ElementPenalty
    penalty::P
    ScaledElementPenalty(pen::P) = typeof(λ) <: Number ? new(pen) : _scaled_error()
end
ScaledElementPenalty{P, λ}(pen::P, ::Type{Val{λ}}) = ScaledElementPenalty{P,λ}(pen)
Base.show{P, λ}(io::IO, sp::ScaledElementPenalty{P, λ}) = println(io, "$λ * ", sp.penalty)

scaled(p::ElementPenalty, λ::Number) = ScaledElementPenalty(p, Val{λ})

value{P, λ}(p::ScaledElementPenalty{P, λ}, θ::Number) = λ * value(p.penalty, θ)
deriv{P, λ}(p::ScaledElementPenalty{P, λ}, θ::Number) = λ * deriv(p.penalty, θ)
prox{P, λ}(p::ScaledElementPenalty{P, λ}, θ::Number) = prox(p.penalty, θ, λ)

# SCAD is special
for f in [:value, :deriv]
    @eval function ($f){P <: SCADPenalty, λ}(p::ScaledElementPenalty{P, λ}, θ::Number)
        ($f)(p.penalty, θ, λ)
    end
end
