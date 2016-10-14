__precompile__(true)

module Penalties

importall LearnBase

export
    Penalty,
        ElementwisePenalty,
            NoPenalty,
            L1Penalty,
            L2Penalty,
            ElasticNetPenalty,
            SCADPenalty,
        ArrayPenalty,
            NuclearNormPenalty,
            GroupLassoPenalty

typealias AA{T, N} AbstractArray{T, N}

abstract ElementwisePenalty <: Penalty
abstract ArrayPenalty <: Penalty


# common functions
soft_thresh{T<:Number}(x::T, λ::T) = sign(x) * max(zero(T), abs(x) - λ)
function soft_thresh!{T<:Number}(x::AA{T}, λ::T)
    for i in eachindex(x)
        @inbounds x[i] = soft_thresh(x[i], λ)
    end
    x
end


include("elementwise.jl")
include("array.jl")
end
