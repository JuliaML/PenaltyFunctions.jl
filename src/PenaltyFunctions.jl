module PenaltyFunctions

using LinearAlgebra
using InteractiveUtils
using RecipesBase

# trait functions
include("traits.jl")

# penalty functions
include("penalties.jl")

# IO and plot recipes
include("printing.jl")
include("plotrecipes.jl")

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
    addgrad, prox, prox!, deriv, value, grad, grad!, addgrad!, scaled

end
