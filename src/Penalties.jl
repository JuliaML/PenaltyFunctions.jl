module Penalties

importall LearnBase

export
    L1Penalty,
    L2Penalty,
    ElasticNetPenalty

#------------------------------------------------------------------# abstract methods
function value{T <: Number}(p::Penalty, λ::T, x::AbstractArray{T})
    result = zero(T)
    for xi in x
        result += value(p, λ, xi)
    end
    result
end

function grad!{T<:Number}(dest::AbstractArray{T}, p::Penalty, λ::T, x::AbstractArray{T})
    @assert size(dest) == size(x)
    for i in eachindex(dest)
        @inbounds dest[i] = deriv(p, λ, x[i])
    end
    dest
end
function grad{T<:Number}(p::Penalty, λ::T, x::AbstractArray{T})
    dest = zeros(x)
    grad!(dest, p, λ, x)
end

function prox!{T<:Number}(p::Penalty, λ::T, x::AbstractArray{T})
    for i in eachindex(x)
        @inbounds x[i] = prox(p, λ, x[i])
    end
    x
end
function prox!{T<:Number}(p::Penalty, λ::AbstractArray{T}, x::AbstractArray{T})
    @assert size(λ) == size(x)
    for i in eachindex(x)
        @inbounds x[i] = prox(p, λ[i], x[i])
    end
    x
end


soft_thresh{T<:Number}(x::T, λ::T) = sign(x) * max(zero(x), abs(x) - λ)
function soft_thresh!{T<:Number}(x::AbstractArray{T}, λ::T)
    for i in eachindex(x)
        @inbounds x[i] = soft_thresh(x[i], λ)
    end
end


#-------------------------------------------------------------------------# L1Penalty
"L1-Norm Penalty: f(x) = vecnorm(x, 1)"
immutable L1Penalty <: Penalty end
value{T<:Number}(p::L1Penalty, λ::T, x::T) = λ * abs(x)
deriv{T<:Number}(p::L1Penalty, λ::T, x::T) = λ * sign(x)
prox{T<:Number}(p::L1Penalty, λ::T, x::T) = λ * soft_thresh(x, λ)

#-------------------------------------------------------------------------# L2Penalty
"Squared L2-Norm Penalty: f(x) = vecnorm(x, 2) ^ 2"
immutable L2Penalty <: Penalty end
value{T<:Number}(p::L2Penalty, λ::T, x::T) = λ * T(0.5) * x ^ 2 # FIXME?
deriv{T<:Number}(p::L2Penalty, λ::T, x::T) = λ * x
prox{T<:Number}(p::L2Penalty, λ::T, x::T) = x / (one(T) + λ)

#-----------------------------------------------------------------# ElasticNetPenalty
"Weighted average of L1Penalty and L2Penalty"
immutable ElasticNetPenalty{T <: Number} <: Penalty
    α::T
end
function ElasticNetPenalty(α::Number = 0.5)
    @assert 0 < α < 1
    ElasticNetPenalty(α)
end
function value{T<:Number}(p::ElasticNetPenalty{T}, λ::T, x::T)
    λ * (value(L1Penalty(), p.α, x) + value(L2Penalty(), one(T) - p.α, x))
end
function deriv{T<:Number}(p::ElasticNetPenalty{T}, λ::T, x::T)
     λ * (deriv(L1Penalty(), p.α, x) + deriv(L2Penalty(), one(T) - p.α, x))
end
function prox{T<:Number}(p::ElasticNetPenalty, λ::T, x::T)
    l1prox = prox(L1Penalty(), p.α * λ, x)
    prox(L2Penalty(), (one(T) - p.α) * λ, l1prox)
end

end
