# Penalties that can be evaluated elementwise

#------------------------------------------------------------------# abstract methods
function value{T <: Number}(p::Penalty, x::AA{T})
    result = zero(T)
    for xi in x
        result += value(p, xi)
    end
    result
end
function value{T <: Number}(p::Penalty, x::AA{T}, ρ::AA{T})
    @assert size(x) == size(ρ)
    result = zero(T)
    for i in eachindex(x)
        @inbounds result += ρ[i] * value(p, x[i])
    end
    result
end


function grad!{T<:Number}(dest::AA{T}, p::Penalty, x::AA{T})
    @assert size(dest) == size(x)
    for i in eachindex(dest)
        @inbounds dest[i] = deriv(p, x[i])
    end
    dest
end
function grad!{T<:Number}(dest::AA{T}, p::Penalty, x::AA{T}, ρ::AA{T})
    @assert size(dest) == size(x) == size(ρ)
    for i in eachindex(dest)
        @inbounds dest[i] = ρ[i] * deriv(p, x[i])
    end
    dest
end


function prox!{T<:Number}(p::Penalty, x::AA{T})
    for i in eachindex(x)
        @inbounds x[i] = prox(p, x[i])
    end
    x
end
function prox!{T<:Number}(p::Penalty, x::AA{T}, ρ::AA{T})
    @assert size(x) == size(ρ)
    for i in eachindex(x)
        @inbounds x[i] = prox(p, x[i], ρ[i])
    end
    x
end


soft_thresh{T<:Number}(x::T, λ::T) = sign(x) * max(zero(T), abs(x) - λ)


prox(p::Penalty, x::Number) = _prox(p, x, p.λ)
prox{T<:Number}(p::Penalty, x::T, ρ::T) = _prox(p, x, p.λ * ρ)


#-------------------------------------------------------------------------# L1Penalty
"L1-Norm Penalty: f(x) = vecnorm(x, 1)"
type L1Penalty{T <: Number} <: Penalty
    λ::T
end
L1Penalty(λ::Number = 0.1) = L1Penalty(λ)
value{T<:Number}(p::L1Penalty{T}, x::T) = p.λ * abs(x)
deriv{T<:Number}(p::L1Penalty{T}, x::T) = p.λ * sign(x)
_prox{T<:Number}(p::L1Penalty{T}, x::T, λ::T) = soft_thresh(x, λ)


#-------------------------------------------------------------------------# L2Penalty
"Squared L2-Norm Penalty: f(x) = vecnorm(x, 2) ^ 2"
type L2Penalty{T <: Number} <: Penalty
    λ::T
end
L2Penalty(λ::Number = 0.1) = L2Penalty(λ)
value{T<:Number}(p::L2Penalty{T}, x::T) = p.λ * T(.5) * x * x # FIXME?
deriv{T<:Number}(p::L2Penalty{T}, x::T) = p.λ * x
_prox{T<:Number}(p::L2Penalty{T}, x::T, λ::T) = x / (one(T) + λ)

#-----------------------------------------------------------------------# ElasticNetPenalty
"Weighted average of L1Penalty and L2Penalty"
type ElasticNetPenalty{T <: Number} <: Penalty
    λ::T
    α::T
end
ElasticNetPenalty(λ::Number = 0.1, α::Number = 0.5) = ElasticNetPenalty(λ, α)
function value{T<:Number}(p::ElasticNetPenalty{T}, x::T)
    p.λ * (p.α * abs(x) + (one(T) - p.α) * T(.5) * x * x) # FIXME?
end
function deriv{T<:Number}(p::ElasticNetPenalty{T}, x::T)
    p.λ * (p.α * sign(x) + (one(T) - p.α) * x)
end
function _prox{T<:Number}(p::ElasticNetPenalty{T}, x::T, ρ::T)
    soft_thresh(x / (one(T) + (one(T) - p.α) * λ), p.α * λ)
end
