__precompile__(true)

module PenaltyFunctions

importall LearnBase
# export LearnBase
eval(Expr(:toplevel, Expr(:export, setdiff(names(LearnBase), [:LearnBase])...)))

using RecipesBase

export
    Penalty,
        ElementPenalty,
            NoPenalty,
            L1Penalty,
            L2Penalty,
            ElasticNetPenalty,
            SCADPenalty,
            MCPPenalty,
            LogPenalty,
        ArrayPenalty,
            NuclearNormPenalty,
            GroupLassoPenalty,
            MahalanobisPenalty,
    addgrad

typealias AA{T, N} AbstractArray{T, N}

const ϵ = 1e-6 # avoid dividing by 0, etc.


# common functions

soft_thresh{T<:Number}(x::T, λ::T) = max(zero(T), x - sign(x) * λ)

function soft_thresh!{T<:Number}(x::AA{T}, λ::T)
    for i in eachindex(x)
        @inbounds x[i] = soft_thresh(x[i], λ)
    end
    x
end

function name(p::Penalty)
    s = replace(string(typeof(p)), "PenaltyFunctions.", "")

end
Base.show(io::IO, p::Penalty) = print(io, name(p))

include("elementpenalty.jl")
include("arraypenalty.jl")
end
