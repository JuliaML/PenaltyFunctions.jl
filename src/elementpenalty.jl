"""
Penalties that are applied element-wise.
"""
abstract ElementPenalty

#-------------------------------------------------------------------------------# methods
value{T<:Number}(p::ElementPenalty, θ::T, s::T)     = s * value(p, θ)
value{T<:Number}(p::ElementPenalty, θ::AA{T})       = sum(x -> value(p, x), θ)
value{T<:Number}(p::ElementPenalty, θ::AA{T}, s::T) = sum(x -> value(p, x, s), θ)
function value{T<:Number}(p::ElementPenalty, θ::AA{T}, s::AA{T})
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
immutable NoPenalty <: ElementPenalty end
value(p::NoPenalty, θ::Number) = zero(θ)
deriv(p::NoPenalty, θ::Number) = zero(θ)
prox{T <: Number}(p::NoPenalty, θ::T, s::T) = θ

immutable L1Penalty <: ElementPenalty end
value(p::L1Penalty, θ::Number) = abs(θ)
deriv(p::L1Penalty, θ::Number) = sign(θ)
prox{T <: Number}(p::L1Penalty, θ::T, s::T) = soft_thresh(θ, s)

immutable L2Penalty <: ElementPenalty end
value(p::L2Penalty, θ::Number) = typeof(θ)(0.5) * θ * θ
deriv(p::L2Penalty, θ::Number) = θ
prox{T<:Number}(p::L2Penalty, θ::T, s::T) = θ / (one(T) + s)

immutable ElasticNetPenalty{T <: Number} <: ElementPenalty α::T end
ElasticNetPenalty(α::Number) = (@assert 0 <= α <= 1; ElasticNetPenalty(α))
function value{T <: Number}(p::ElasticNetPenalty{T}, θ::T)
    p.α * value(L1Penalty(), θ) + (1 - p.α) * value(L2Penalty(), θ)
end
function deriv{T <: Number}(p::ElasticNetPenalty{T}, θ::T)
    p.α * deriv(L1Penalty(), θ) + (1 - p.α) * deriv(L2Penalty(), θ)
end
function prox{T <: Number}(p::ElasticNetPenalty{T}, θ::T, s::T)
    αs = p.α * s
    soft_thresh(θ, αs) / (one(T) + s - αs)
end

# #-----------------------------------------------------------------------# SCADPenalty
# # http://www.pstat.ucsb.edu/student%20seminar%20doc/SCAD%20Jian%20Shi.pdf
# # For prox: http://arxiv.org/pdf/1412.2999.pdf
# # Needs tests
# type SCADPenalty{T <: Number} <: ElementPenalty
#     λ::T
#     a::T
# end
# function SCADPenalty(λ::Number = 0.1, a::Number = 3.7)
#     @assert λ >= zero(λ)
#     @assert a > T(2)
#     SCADPenalty(λ, a)
# end
# function value{T <: Number}(p::SCADPenalty{T}, θi::T)
#     absθ = abs(θi)
#     a, λ = p.a, p.λ
#     if absθ < λ
#         return λ * absθ
#     elseif absθ <= λ * a
#         return -T(0.5) * (absθ ^ 2 - T(2) * a * λ * absθ + λ ^ 2) / (a - one(T))
#     else
#         return T(0.5) * (a + one(T)) * λ * λ
#     end
# end
# function deriv{T <: Number}(p::SCADPenalty{T}, θi::T)
#     absθ = abs(θi)
#     a, λ = p.a, p.λ
#     if absθ < λ
#         return λ * sign(θi)
#     elseif absθ <= λ * a
#         return -(absθ - a * λ * sign(θi)) / (a - one(T))
#     else
#         return zero(T)
#     end
# end
# function _prox{T <: Number}(p::SCADPenalty{T}, θi::T, λ::T)
#     a = p.a
#     absθ = abs(θi)
#     if absθ < T(2) * λ
#         return soft_thresh(θi, λ)
#     elseif absθ <= λ * a
#         return (θi - λ * sign(θi) * a / (a - one(T))) / (one(T) - one(T) / (a - one(T)))
#     else
#         return θi
#     end
# end
