module Penalties

using Reexport
importall LearnBase
@reexport using LearnBase

export
    L1Penalty

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


soft_thresh{T<:Number}(x::T, λ::T) = sign(x) * max(zero(x), abs(x) - λ)
function soft_thresh!{T<:Number}(x::AbstractArray{T}, λ::T)
    for i in eachindex(x)
        @inbounds x[i] = soft_thresh(x[i], λ)
    end
end

#-------------------------------------------------------------------------# L1Penalty
immutable L1Penalty <: Penalty end
value{T<:Number}(p::L1Penalty, λ::T, x::T) = λ * abs(x)
deriv{T<:Number}(p::L1Penalty, λ::T, x::T) = λ * sign(x)
prox{T<:Number}(p::L1Penalty, λ::T, x::T) = λ * soft_thresh(x, λ)

end
