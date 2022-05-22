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

name(p::ArrayPenalty) = replace(string(typeof(p)), "PenaltyFunctions." => "")

Base.show(io::IO, p::Penalty) = print(io, name(p))
Base.show(io::IO, sp::ScaledArrayPenalty) = print(io, "$(sp.λ) * ($(sp.penalty))")
