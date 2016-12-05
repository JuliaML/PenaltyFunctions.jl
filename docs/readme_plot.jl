module ReadmePlot

using PenaltyFunctions, Plots; pyplot()

λ = .1
p = plot(L1Penalty(λ))
for pen in [L2Penalty(λ), ElasticNetPenalty(λ), SCADPenalty(λ)]
    plot!(pen)
end
png(p, Pkg.dir("PenaltyFunctions", "docs", "readmefig.png"))

end
