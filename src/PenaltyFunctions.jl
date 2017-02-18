__precompile__(true)

module PenaltyFunctions

importall LearnBase
# export LearnBase
eval(Expr(:toplevel, Expr(:export, setdiff(names(LearnBase), [:LearnBase])...)))

using RecipesBase

export
    Penalty,
        ElementPenalty,
            ConvexElementPenalty,
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


# common functions
soft_thresh{T<:Number}(x::T, λ::T) = sign(x) * max(zero(T), abs(x) - λ)

function soft_thresh!{T<:Number}(x::AA{T}, λ::T)
    for i in eachindex(x)
        @inbounds x[i] = soft_thresh(x[i], λ)
    end
    x
end

function name(p::Penalty)
    s = replace(string(typeof(p)), "PenaltyFunctions.", "")
    # s = replace(s, r"\{(.*)", "")
    f = fieldnames(p)
    flength = length(f)
    if flength > 0
        s *= "("
        for (i, field) in enumerate(f)
            s *= "$field = $(getfield(p, field))"
            if i < flength
                s *= ", "
            end
        end
        s *= ")"
    end
    s
end
Base.show(io::IO, p::Penalty) = print(io, name(p))

include("elementpenalty.jl")
include("arraypenalty.jl")

end
