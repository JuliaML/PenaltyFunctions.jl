module ReadmePlots

using PenaltyFunctions, LearnBase, LaTeXStrings, Plots; pyplot(palette = :darktest)

# value for univariate parameter

p = plot(NoPenalty(), grid=false, ylim = (0, 5))
for pen in [L1Penalty(), L2Penalty(), ElasticNetPenalty(.5), SCADPenalty(3.7),
            MCPPenalty(1.), LogPenalty(1.)]
    plot!(pen)
end
plot!(xlab = L"\theta", ylab = L"g(\theta)")

p[1][1][:linestyle] = :dot
p[1][2][:linestyle] = :dash
p[1][3][:linestyle] = :line
p[1][4][:linestyle] = :dot
p[1][5][:linestyle] = :dash
p[1][6][:linestyle] = :line
p[1][7][:linestyle] = :dot

display(p)
savefig(p, Pkg.dir("PenaltyFunctions", "docs", "readmefig.png"))

end
