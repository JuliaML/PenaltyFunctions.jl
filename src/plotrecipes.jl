@recipe function f(p::ElementPenalty)
    label --> name(p)
    xlims --> (-4, 4)
    ylabel --> "penalty value"
    xlabel --> "parameter"
    g(x) = value(p, x, 1.)
    g
end