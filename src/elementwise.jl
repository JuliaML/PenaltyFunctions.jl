# Penalties that can be evaluated elementwise

#------------------------------------------------------------------# abstract methods
function value{T <: Number}(p::ElementwisePenalty, x::AA{T})
    result = zero(T)
    for xi in x
        result += value(p, xi)
    end
    result
end
value{T <: Number}(p::ElementwisePenalty, x::AA{T}, s::T) = value(p, x) * s
function value{T <: Number}(p::ElementwisePenalty, x::AA{T}, s::AA{T})
    @assert size(x) == size(s)
    result = zero(T)
    for i in eachindex(x)
        @inbounds result += value(p, x[i], s[i])
    end
    result
end


function grad!{T<:Number}(dest::AA{T}, p::ElementwisePenalty, x::AA{T})
    @assert size(dest) == size(x)
    for i in eachindex(dest)
        @inbounds dest[i] = deriv(p, x[i])
    end
    dest
end
function grad!{T<:Number}(dest::AA{T}, p::ElementwisePenalty, x::AA{T}, s::T)
    @assert size(dest) == size(x)
    for i in eachindex(dest)
        @inbounds dest[i] = deriv(p, x[i], s)
    end
    dest
end
function grad!{T<:Number}(dest::AA{T}, p::ElementwisePenalty, x::AA{T}, s::AA{T})
    @assert size(dest) == size(x) == size(s)
    for i in eachindex(dest)
        @inbounds dest[i] = deriv(p, x[i], s[i])
    end
    dest
end


function prox!{T<:Number}(p::ElementwisePenalty, x::AA{T})
    for i in eachindex(x)
        @inbounds x[i] = prox(p, x[i])
    end
    x
end
function prox!{T<:Number}(p::ElementwisePenalty, x::AA{T}, s::T)
    for i in eachindex(x)
        @inbounds x[i] = prox(p, x[i], s)
    end
    x
end
function prox!{T<:Number}(p::ElementwisePenalty, x::AA{T}, s::AA{T})
    @assert size(x) == size(s)
    for i in eachindex(x)
        @inbounds x[i] = prox(p, x[i], s[i])
    end
    x
end


soft_thresh{T<:Number}(x::T, λ::T) = sign(x) * max(zero(T), abs(x) - λ)
function soft_thresh!{T<:Number}(x::AA{T}, λ::T)
    for i in eachindex(x)
        @inbounds x[i] = soft_thresh(x[i], λ)
    end
    x
end


value{T<:Number}(p::ElementwisePenalty, x::T, s::T) = s * value(p, x)
deriv{T<:Number}(p::ElementwisePenalty, x::T, s::T) = s * deriv(p, x)
prox(p::ElementwisePenalty, x::Number) = _prox(p, x, p.λ)
prox{T<:Number}(p::ElementwisePenalty, x::T, s::T) = _prox(p, x, p.λ * s)


#-------------------------------------------------------------------------# NoPenalty
"f(x) = 0"
type NoPenalty <: ElementwisePenalty end
value(p::NoPenalty, x::Number) = zero(x)
deriv(p::NoPenalty, x::Number) = zero(x)
prox{T<:Number}(p::NoPenalty, x::T) = x
prox{T<:Number}(p::NoPenalty, x::T, s::T) = x


#-------------------------------------------------------------------------# L1Penalty
"L1-Norm Penalty: f(x) = vecnorm(x, 1)"
type L1Penalty{T <: Number} <: ElementwisePenalty
    λ::T
end
function L1Penalty(λ::Number = 0.1)
    @assert λ >= zero(λ)
    L1Penalty(λ)
end
value{T<:Number}(p::L1Penalty{T}, x::T) = p.λ * abs(x)
deriv{T<:Number}(p::L1Penalty{T}, x::T) = p.λ * sign(x)
_prox{T<:Number}(p::L1Penalty{T}, x::T, λ::T) = soft_thresh(x, λ)


#-------------------------------------------------------------------------# L2Penalty
"Squared L2-Norm Penalty: f(x) = vecnorm(x, 2) ^ 2"
type L2Penalty{T <: Number} <: ElementwisePenalty
    λ::T
end
function L2Penalty(λ::Number = 0.1)
    @assert λ >= zero(λ)
    L2Penalty(λ)
end
value{T<:Number}(p::L2Penalty{T}, x::T) = p.λ * T(.5) * x * x
deriv{T<:Number}(p::L2Penalty{T}, x::T) = p.λ * x
_prox{T<:Number}(p::L2Penalty{T}, x::T, λ::T) = x / (one(T) + λ)

#-----------------------------------------------------------------# ElasticNetPenalty
# λ₁ = λ * α, λ₂ = λ * (1 - α)
"Weighted average of L1Penalty and L2Penalty"
type ElasticNetPenalty{T <: Number} <: ElementwisePenalty
    λ::T
    α::T
end
function ElasticNetPenalty(λ::Number = 0.1, α::Number = 0.5)
    @assert λ >= zero(λ)
    @assert zero(λ) <= α <= one(λ)
    ElasticNetPenalty(λ, α)
end
function value{T<:Number}(p::ElasticNetPenalty{T}, x::T)
    p.λ * (p.α * abs(x) + (one(T) - p.α) * T(.5) * x * x)
end
function deriv{T<:Number}(p::ElasticNetPenalty{T}, x::T)
    p.λ * (p.α * sign(x) + (one(T) - p.α) * x)
end
function _prox{T<:Number}(p::ElasticNetPenalty{T}, x::T, λ::T)
    αλ = p.α * λ
    soft_thresh(x, αλ) / (one(T) + λ - αλ)
end


#--------------------------------------------------------------# HardThresholdPenalty
# Needs tests
type HardThresholdPenalty{T <: Number} <: ElementwisePenalty
    λ::T
end
function HardThresholdPenalty(λ::Number = 0.1)
    @assert λ >= zero(λ)
    HardThresholdPenalty(λ)
end
function value{T <: Number}(p::HardThresholdPenalty{T}, x::T)
    absx = abs(x)
    λ = p.λ
    T(.5) * λ * λ - (absx - λ) ^ 2 * T(absx < λ)
end
function deriv{T <: Number}(p::HardThresholdPenalty{T}, x::T)
    absx = abs(x)
    λ = p.λ
    (λ - absx) * T(absx > λ)
end

function _prox{T <: Number}(p::HardThresholdPenalty{T}, x::T, λ::T)
    if x > p.λ
        return x
    else
        return zero(T)
    end
end


#-----------------------------------------------------------------------# SCADPenalty
# http://www.pstat.ucsb.edu/student%20seminar%20doc/SCAD%20Jian%20Shi.pdf
# For prox: http://arxiv.org/pdf/1412.2999.pdf
# Needs tests
type SCADPenalty{T <: Number} <: ElementwisePenalty
    λ::T
    a::T
end
function SCADPenalty(λ::Number = 0.1, a::Number = 3.7)
    @assert λ >= zero(λ)
    @assert a > T(2)
    SCADPenalty(λ, a)
end
function value{T <: Number}(p::SCADPenalty{T}, x::T)
    abx = abs(x)
    a, λ = p.a, p.λ
    if abx < λ
        return λ * abx
    elseif abx <= λ * a
        return -T(0.5) * (abx ^ 2 - T(2) * a * λ * abx + λ ^ 2) / (a - one(T))
    else
        return T(0.5) * (a + one(T)) * λ * λ
    end
end
function deriv{T <: Number}(p::SCADPenalty{T}, x::T)
    abx = abs(x)
    a, λ = p.a, p.λ
    if abx < λ
        return λ * sign(x)
    elseif abx <= λ * a
        return -(abx - a * λ * sign(x)) / (a - one(T))
    else
        return zero(T)
    end
end
function _prox{T <: Number}(p::SCADPenalty{T}, x::T, λ::T)
    a = p.a
    abx = abs(x)
    if abx < T(2) * λ
        return soft_thresh(x, λ)
    elseif abx <= λ * a
        return (x - λ * sign(x) * a / (a - one(T))) / (one(T) - one(T) / (a - one(T)))
    else
        return x
    end
end
