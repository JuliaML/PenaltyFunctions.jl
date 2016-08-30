# Penalties that evaluate on the entire array only

#----------------------------------------------------------------# NuclearNormPenalty
# needs tests
type NuclearNormPenalty{T <: Number} <: ArrayPenalty
    λ::T
end
NuclearNormPenalty(λ::Number = 0.1) = NuclearNormPenalty(λ)

function value{T <: Number}(p::NuclearNormPenalty{T}, A::AA{T, 2})
    if size(A,1) > size(A,2)
        return trace(sqrtm(A'A))
    else
        return trace(sqrtm(A * A'))
    end
end

function prox!{T <: Number}(A::AA{T, 2}, p::NuclearNormPenalty{T}, s::T)
    svdecomp = svdfact!(A)
    soft_thresh!(svdecomp.S, s * p.λ)
    copy!(A, full(svdecomp))
end

function prox{T <: Number}(A::AA{T, 2}, p::NuclearNormPenalty{T}, s::T)
    svdecomp = svdfact(A)
    soft_thresh!(svdecomp.S, s * p.λ)
    full(svdecomp)
end


#--------------------------------------------------------------# GeneralizedL1Penalty
# http://www.stat.cmu.edu/~ryantibs/papers/genlasso.pdf
# TODO
