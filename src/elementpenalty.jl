# Penalties that can be evaluated elementwise

# #------------------------------------------------------------------# abstract methods
# function value{T <: Number}(p::ElementPenalty, θ::AA{T})
#     result = zero(T)
#     for θi in θ
#         result += value(p, θi)
#     end
#     result
# end
# value{T <: Number}(p::ElementPenalty, θ::AA{T}, s::T) = value(p, θ) * s
# function value{T <: Number}(p::ElementPenalty, θ::AA{T}, s::AA{T})
#     @assert size(θ) == size(s)
#     result = zero(T)
#     for i in eachindex(θ)
#         @inbounds result += value(p, θ[i], s[i])
#     end
#     result
# end
#
#
# function grad!{T<:Number}(dest::AA{T}, p::ElementPenalty, θ::AA{T})
#     @assert size(dest) == size(θ)
#     for i in eachindex(dest)
#         @inbounds dest[i] = deriv(p, θ[i])
#     end
#     dest
# end
# function grad!{T<:Number}(dest::AA{T}, p::ElementPenalty, θ::AA{T}, s::T)
#     @assert size(dest) == size(θ)
#     for i in eachindex(dest)
#         @inbounds dest[i] = deriv(p, θ[i], s)
#     end
#     dest
# end
# function grad!{T<:Number}(dest::AA{T}, p::ElementPenalty, θ::AA{T}, s::AA{T})
#     @assert size(dest) == size(θ) == size(s)
#     for i in eachindex(dest)
#         @inbounds dest[i] = deriv(p, θ[i], s[i])
#     end
#     dest
# end
#
# function addgrad!{T<:Number}(∇::AA{T}, penalty::ElementPenalty, θ::AA{T})
#     @inbounds for (i, θi) in zip(eachindex(∇), θ)
#         ∇[i] += deriv(penalty, θi)
#     end
#     ∇
# end
# function addgrad!{T<:Number}(∇::AA{T}, penalty::ElementPenalty, θ::AA{T}, s::T)
#     @inbounds for (i, θi) in zip(eachindex(∇), θ)
#         ∇[i] += deriv(penalty, θi, s)
#     end
#     ∇
# end
# function addgrad!{T<:Number}(∇::AA{T}, penalty::ElementPenalty, θ::AA{T}, s::AA{T})
#     @assert size(∇) == size(θ) == size(s)
#     @inbounds for i in eachindex(∇)
#         ∇[i] += deriv(penalty, θ[i], s[i])
#     end
#     ∇
# end
#
#
# function prox!{T<:Number}(p::ElementPenalty, θ::AA{T})
#     for i in eachindex(θ)
#         @inbounds θ[i] = prox(p, θ[i])
#     end
#     θ
# end
# function prox!{T<:Number}(p::ElementPenalty, θ::AA{T}, s::T)
#     for i in eachindex(θ)
#         @inbounds θ[i] = prox(p, θ[i], s)
#     end
#     θ
# end
# function prox!{T<:Number}(p::ElementPenalty, θ::AA{T}, s::AA{T})
#     @assert size(θ) == size(s)
#     for i in eachindex(θ)
#         @inbounds θ[i] = prox(p, θ[i], s[i])
#     end
#     θ
# end

#-------------------------------------------------------------------------------# methods
value{T<:Number}(p::ElementPenalty, θ::AA{T}) = sum(x -> value(p, x), θ)

prox!{T}(p::ElementPenalty, θ::AA{T}, s::T) = map!(θj -> prox(p, θj, s), θ)

prox!{T}(p::ElementPenalty, θ::AA{T}, s::AA{T}) = map!((θj, sj) -> prox(p, θj, sj), θ, s)



#----------------------------------------------------------------------# ElementPenalties
immutable NoPenalty <: ElementPenalty end
value(p::NoPenalty, θ::Number) = zero(typeof(θ))
deriv(p::NoPenalty, θ::Number) = zero(typeof(θ))
prox{T <: Number}(p::NoPenalty, θ::T, s::T) = θ

immutable L1Penalty <: ElementPenalty end
value(p::L1Penalty, θ::Number) = abs(θ)
deriv(p::L1Penalty, θ::Number) = sign(θ)
prox{T <: Number}(p::L1Penalty, θ::T, s::T) = soft_thresh(θ, s)

immutable L2Penalty <: ElementPenalty end
value(p::L2Penalty, θ::Number) = typeof(θ)(0.5) * θ * θ
deriv(p::L2Penalty, θ::Number) = θ
prox{T<:Number}(p::L2Penalty, θ::T, λ::T) = θ / (one(T) + λ)

immutable ElasticNetPenalty{T <: Number} <: ElementPenalty α::T end
ElasticNetPenalty(α::Number) = (@assert 0 <= α <= 1; ElasticNetPenalty(α))
function value{T <: Number}(p::ElasticNetPenalty{T}, θ::T)
    p.α * value(L1Penalty(), θ) + (1 - p.α) * value(L2Penalty(), θ)
end
function deriv{T <: Number}(p::ElasticNetPenalty{T}, θ::T)
    p.α * deriv(L1Penalty(), θ) + (1 - p.α) * deriv(L2Penalty(), θ)
end
function prox{T <: Number}(p::ElasticNetPenalty{T}, θ::T, λ::T)
    αλ = p.α * λ
    soft_thresh(θ, αλ) / (one(T) + λ - αλ)
end

# #-------------------------------------------------------------------------# NoPenalty
# "No Penalty: g(θ) = 0"
# type NoPenalty <: ElementPenalty end
# name(p::NoPenalty) = "NoPenalty"
# value(p::NoPenalty, θi::Number) = zero(θi)
# deriv(p::NoPenalty, θi::Number) = zero(θi)
# prox{T<:Number}(p::NoPenalty, θi::T) = θi
# prox{T<:Number}(p::NoPenalty, θi::T, s::T) = θi
#
#
# #-------------------------------------------------------------------------# L1Penalty
# "L1-Norm Penalty: g(θ) = vecnorm(θ, 1)"
# type L1Penalty{T <: Number} <: ElementPenalty
#     λ::T
# end
# function L1Penalty(λ::Number = 0.1)
#     @assert λ >= zero(λ)
#     L1Penalty(λ)
# end
# value{T<:Number}(p::L1Penalty{T}, θi::T) = p.λ * abs(θi)
# deriv{T<:Number}(p::L1Penalty{T}, θi::T) = p.λ * sign(θi)
# _prox{T<:Number}(p::L1Penalty{T}, θi::T, λ::T) = soft_thresh(θi, λ)
#
#
# #-------------------------------------------------------------------------# L2Penalty
# "Squared L2-Norm Penalty: g(θ) = 0.5 * vecnorm(θ, 2) ^ 2"
# type L2Penalty{T <: Number} <: ElementPenalty
#     λ::T
# end
# function L2Penalty(λ::Number = 0.1)
#     @assert λ >= zero(λ)
#     L2Penalty(λ)
# end
# value{T<:Number}(p::L2Penalty{T}, θi::T) = p.λ * T(.5) * θi * θi
# deriv{T<:Number}(p::L2Penalty{T}, θi::T) = p.λ * θi
# _prox{T<:Number}(p::L2Penalty{T}, θi::T, λ::T) = θi / (one(T) + λ)
#
# #-----------------------------------------------------------------# ElasticNetPenalty
# # λ₁ = λ * α, λ₂ = λ * (1 - α)
# "Weighted average of L1Penalty and L2Penalty"
# type ElasticNetPenalty{T <: Number} <: ElementPenalty
#     λ::T
#     α::T
# end
# function ElasticNetPenalty(λ::Number = 0.1, α::Number = 0.5)
#     @assert λ >= zero(λ)
#     @assert zero(λ) <= α <= one(λ)
#     ElasticNetPenalty(λ, α)
# end
# function value{T<:Number}(p::ElasticNetPenalty{T}, θi::T)
#     p.λ * (p.α * abs(θi) + (one(T) - p.α) * T(.5) * θi * θi)
# end
# function deriv{T<:Number}(p::ElasticNetPenalty{T}, θi::T)
#     p.λ * (p.α * sign(θi) + (one(T) - p.α) * θi)
# end
# function _prox{T<:Number}(p::ElasticNetPenalty{T}, θi::T, λ::T)
#     αλ = p.α * λ
#     soft_thresh(θi, αλ) / (one(T) + λ - αλ)
# end
#
#
#
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
