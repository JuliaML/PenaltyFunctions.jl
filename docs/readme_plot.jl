module ReadmePlots

using PenaltyFunctions, LearnBase, Plots; pyplot(palette = :darktest)

# value for univariate parameter
λ = .1
p = plot(L1Penalty(λ))
for pen in [L2Penalty(λ), ElasticNetPenalty(λ), SCADPenalty(λ)]
    plot!(pen)
end
savefig(p, Pkg.dir("PenaltyFunctions", "docs", "readmefig.png"))

end
