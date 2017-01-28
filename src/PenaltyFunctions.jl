__precompile__(true)

module PenaltyFunctions

importall LearnBase
using RecipesBase

export
    Penalty,
        ElementPenalty,
            NoPenalty,
            L1Penalty,
            L2Penalty,
            ElasticNetPenalty,
            SCADPenalty,
        ArrayPenalty,
            NuclearNormPenalty,
            GroupLassoPenalty,
            MahalanobisPenalty

typealias AA{T, N} AbstractArray{T, N}

abstract ElementPenalty <: Penalty
abstract ArrayPenalty <: Penalty


# common functions
soft_thresh{T<:Number}(x::T, λ::T) = max(zero(T), x - sign(x) * λ)

function soft_thresh!{T<:Number}(x::AA{T}, λ::T)
    for i in eachindex(x)
        @inbounds x[i] = soft_thresh(x[i], λ)
    end
    x
end

function name(p::Penalty)
    s = replace(string(p), "PenaltyFunctions.", "")
    s = replace(s, r"\{.+", "")
    s * "(lambda = $(p.λ))"
end

include("elementpenalty.jl")
include("arraypenalty.jl")
end
