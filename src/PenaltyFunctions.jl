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
            MahalanobisPenalty,
    addgrad

typealias AA{T, N} AbstractArray{T, N}


# common functions
soft_thresh{T<:Number}(x::T, λ::T) = max(zero(T), x - sign(x) * λ)

function soft_thresh!{T<:Number}(x::AA{T}, λ::T)
    for i in eachindex(x)
        @inbounds x[i] = soft_thresh(x[i], λ)
    end
    x
end

name(p::Penalty) = replace(string(typeof(p)), "PenaltyFunctions.", "")
Base.show(io::IO, p::Penalty) = print(io, name(p))

include("elementpenalty.jl")
include("arraypenalty.jl")
end
