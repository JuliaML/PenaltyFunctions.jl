module Penalties

importall LearnBase

export
    Penalty,
        ElementwisePenalty,
            NoPenalty,
            L1Penalty,
            L2Penalty,
            ElasticNetPenalty,
            HardThresholdPenalty,
            SCADPenalty,
        ArrayPenalty,
            NuclearNormPenalty

typealias AA{T, N} AbstractArray{T, N}

abstract ElementwisePenalty <: Penalty
abstract ArrayPenalty <: Penalty

include("elementwise.jl")
include("array.jl")
end
