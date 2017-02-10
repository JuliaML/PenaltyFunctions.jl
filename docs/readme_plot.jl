module ReadmePlots

using PenaltyFunctions, LearnBase, Plots; pyplot(palette = :darktest)

# value for univariate parameter
p = plot(NoPenalty(), grid=false, ylim = (0, 5))
for pen in [L1Penalty(), L2Penalty(), ElasticNetPenalty(.5), SCADPenalty(3.7),
            MCPPenalty(1.), LogPenalty(1.)]
    plot!(pen)
end
display(p)
savefig(p, Pkg.dir("PenaltyFunctions", "docs", "readmefig.png"))

end
