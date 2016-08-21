module Penalties

importall LearnBase

export
    L1Penalty,
    L2Penalty,
    ElasticNetPenalty,
    SCADPenalty

typealias AA{T} AbstractArray{T}

include("elementwise.jl")
include("matrix.jl")
end
