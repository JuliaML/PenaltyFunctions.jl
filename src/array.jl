# Penalties that evaluate on the entire array only

prox!{T <: Number}(p::ArrayPenalty, A::AA{T}) = _prox!(p, A, p.λ)
prox!{T <: Number}(p::ArrayPenalty, A::AA{T}, s::T) = _prox!(p, A, s * p.λ)
prox{T <: Number}(p::ArrayPenalty, A::AA{T}) = prox!(p, deepcopy(A))
prox{T <: Number}(p::ArrayPenalty, A::AA{T}, s::T) = prox!(p, deepcopy(A), s)


#----------------------------------------------------------------# NuclearNormPenalty
# needs tests
type NuclearNormPenalty{T <: Number} <: ArrayPenalty
    λ::T
end
NuclearNormPenalty(λ::Number = 0.1) = NuclearNormPenalty(λ)

function value{T <: Number}(p::NuclearNormPenalty{T}, A::AA{T, 2})
    if size(A, 1) > size(A, 2)
        return trace(sqrtm(A'A))
    else
        return trace(sqrtm(A * A'))
    end
end

function _prox!{T <: Number}(p::NuclearNormPenalty{T}, A::AA{T, 2}, s::T)
    svdecomp = svdfact!(A)
    soft_thresh!(svdecomp.S, s)
    copy!(A, full(svdecomp))
end



#--------------------------------------------------------------# GeneralizedL1Penalty
# http://www.stat.cmu.edu/~ryantibs/papers/genlasso.pdf
# TODO
