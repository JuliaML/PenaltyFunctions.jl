module Penalties

importall LearnBase

export
    NoPenalty,
    L1Penalty,
    L2Penalty,
    ElasticNetPenalty,
    SCADPenalty

typealias AA{T} AbstractArray{T}

include("elementwise.jl")
include("matrix.jl")
end
