"""
Unpenalized

`g(θ) = 0`
"""
struct NoPenalty <: ProxableElementPenalty end
value(p::NoPenalty, θ::Number) = zero(θ)
deriv(p::NoPenalty, θ::Number) = zero(θ)
prox(p::NoPenalty,  θ::Number, s::Number) = θ


"""
L1Penalty aka LASSO

`g(θ) = abs(θ)`
"""
struct L1Penalty <: ProxableElementPenalty end
value(p::L1Penalty, θ::Number) = abs(θ)
deriv(p::L1Penalty, θ::Number) = sign(θ)
prox(p::L1Penalty,  θ::Number, s::Number) = soft_thresh(θ, s)


"""
L2Penalty aka Ridge

`g(θ) = .5 * θ ^ 2`
"""
struct L2Penalty <: ProxableElementPenalty end
value(p::L2Penalty, θ::T) where {T <: Number} = inv(T(2)) * θ * θ
deriv(p::L2Penalty, θ::Number) = θ
prox(p::L2Penalty, θ::Number, s::Number) = θ / (one(θ) + s)


"""
ElasticNetPenalty, weighted average of L1Penalty and L2Penalty

`g(θ) = α * abs(θ) + (1 - α) * .5 * θ ^ 2`
"""
struct ElasticNetPenalty{T <: Number} <: ProxableElementPenalty
    α::T
    function ElasticNetPenalty(α::T = 0.5) where {T <: Number}
        0 <= α <= 1 || throw(ArgumentError("α must be in [0, 1]"))
        new{T}(α)
    end
end
for f in (:value, :deriv)
    @eval function ($f)(p::ElasticNetPenalty{T}, θ::Number) where {T <: Number}
        p.α * ($f)(L1Penalty(), θ) + (one(T) - p.α) * ($f)(L2Penalty(), θ)
    end
end
function prox(p::ElasticNetPenalty{T}, θ::Number, s::Number) where {T <: Number}
    αs = p.α * s
    soft_thresh(θ, αs) / (one(T) + s - αs)
end


"""
LogPenalty(η)

`g(θ) = log(1 + η * θ)`
"""
struct LogPenalty{T <: Number} <: ElementPenalty
    η::T
    function LogPenalty(η::T = 1.0) where {T <: Number}
        η > 0 || throw(ArgumentError("η must be > 0"))
        new{T}(η)
    end
end
value(p::LogPenalty, θ::Number) = log1p(p.η * abs(θ))
deriv(p::LogPenalty{T}, θ::Number) where {T <: Number} = p.η * sign(θ) / (one(T) + p.η * abs(θ)) 



# http://www.pstat.ucsb.edu/student%20seminar%20doc/SCAD%20Jian%20Shi.pdf
"""
Smoothly Clipped Absolute Deviation Penalty
"""
struct SCADPenalty{T <: Number} <: ElementPenalty
    a::T
    γ::T
    function SCADPenalty{T}(a::T, γ::T) where T<:Number
        a > 2 || throw(ArgumentError("First parameter must be > 2"))
        γ > 0 || throw(ArgumentError("Second parameter must be > 0"))
        new{T}(a, γ)
    end
end
SCADPenalty(a::T = 3.7, γ::T = T(1)) where {T <: Number} = SCADPenalty{T}(a, γ)
SCADPenalty(a::Number, γ::Number) = SCADPenalty(promote(a, γ)...)

function value(p::SCADPenalty{T}, θ::S) where {T, S <: Number}
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
function deriv(p::SCADPenalty{T}, θ::S) where {T, S <: Number}
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
struct MCPPenalty{T <: Number} <: ElementPenalty
    γ::T  # In paper, this is λ * γ
    function MCPPenalty(γ::T = 2.0) where T
        γ > 0 || throw(ArgumentError("γ must be > 0"))
        new{T}(γ)
    end
end
MCPPenalty(γ::Integer) = MCPPenalty(Float64(γ))
function value(p::MCPPenalty{T}, θ::Number) where {T <: Number}
    t = abs(θ)
    t < p.γ ? t - t^2 / (T(2) * p.γ) : (T(1)/T(2)) * p.γ
end
function deriv(p::MCPPenalty{T}, θ::S) where {T <: Number, S <: Number}
    t = abs(θ)
    t < p.γ ? sign(θ) * (T(1) - t / p.γ) : zero(float(promote_type(S,T)))
end

function _scale_check(λ)
    isa(λ, Number) || throw(ArgumentError("Scale factor λ must be a Number"))
    λ >= 0 || throw(ArgumentError("Scale factor λ has to be strictly positive."))
end

struct ScaledElementPenalty{T <: Number, P <: ElementPenalty} <: ElementPenalty
    penalty::P
    λ::T
end
scaled(p::ElementPenalty, λ::Number) = (_scale_check(λ); ScaledElementPenalty(p, λ))
Base.show(io::IO, sp::ScaledElementPenalty) = print(io, "$(sp.λ) * ($(sp.penalty))")

Base.:(*)(λ::Number, p::ElementPenalty) = scaled(p, λ)

value(p::ScaledElementPenalty{<:Number}, θ::Number) = p.λ * value(p.penalty, θ)
deriv(p::ScaledElementPenalty{<:Number}, θ::Number) = p.λ * deriv(p.penalty, θ)
prox(p::ScaledElementPenalty{<:Number},  θ::Number) = prox(p.penalty, θ, p.λ)
prox(p::ScaledElementPenalty{<:Number},  θ::AbstractArray{<:Number}) = prox(p.penalty, θ, p.λ)

# SCAD is special
for f in (:value, :deriv)
    @eval function ($f)(p::ScaledElementPenalty{<:Number, <:SCADPenalty}, θ::Number)
        ($f)(p.penalty, θ, p.λ)
    end
end
