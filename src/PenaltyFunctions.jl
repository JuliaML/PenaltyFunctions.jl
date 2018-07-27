__precompile__(true)

module PenaltyFunctions

using LearnBase, LinearAlgebra
eval(Expr(:toplevel, Expr(:export, setdiff(names(LearnBase), [:LearnBase])...)))

using RecipesBase

export
    Penalty,
        ElementPenalty,
            ProxableElementPenalty,
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

const AA{T, N} = AbstractArray{T, N}


# common functions
soft_thresh(x::Number, λ::Number) = sign(x) * max(zero(x), abs(x) - λ)

function soft_thresh!(x::AA{<:Number}, λ::Number)
    for i in eachindex(x)
        @inbounds x[i] = soft_thresh(x[i], λ)
    end
    x
end

function name(p::Penalty)
    s = replace(string(typeof(p)), "PenaltyFunctions." => "")
    # s = replace(s, r"\{(.*)", "")
    f = fieldnames(typeof(p))
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
