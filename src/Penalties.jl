module Penalties

importall LearnBase

export
    L1Penalty,
    L2Penalty,
    ElasticNetPenalty

#------------------------------------------------------------------# abstract methods
function value{T <: Number}(p::Penalty, x::AbstractArray{T})
    result = zero(T)
    for xi in x
        result += value(p, xi)
    end
    result
end

function grad!{T<:Number}(dest::AbstractArray{T}, p::Penalty, x::AbstractArray{T})
    @assert size(dest) == size(x)
    for i in eachindex(dest)
        @inbounds dest[i] = deriv(p, x[i])
    end
    dest
end
function grad{T<:Number}(p::Penalty, x::AbstractArray{T})
    dest = zeros(x)
    grad!(dest, p, x)
end

function prox!{T<:Number}(p::Penalty, x::AbstractArray{T})
    for i in eachindex(x)
        @inbounds x[i] = prox(p, x[i])
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
type L1Penalty{T <: Number} <: Penalty
    λ::T
end
L1Penalty(λ::Number = 0.1) = L1Penalty(λ)
value{T<:Number}(p::L1Penalty, x::T) = p.λ * abs(x)
deriv{T<:Number}(p::L1Penalty, x::T) = p.λ * sign(x)
prox{T<:Number}(p::L1Penalty, x::T) = soft_thresh(x, p.λ)



end
