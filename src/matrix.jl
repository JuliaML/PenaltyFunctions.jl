# Penalties that evaluate on a matrix

#----------------------------------------------------------------# NuclearNormPenalty
type NuclearNormPenalty{T <: Number} <: Penalty
    λ::T
end
NuclearNormPenalty(λ::Number = 0.1) = NuclearNormPenalty(λ)

function value{T <: Number}(p::NuclearNormPenalty{T}, A::AbstractMatrix{T})
    if size(A,1) > size(A,2)
        return trace(sqrtm(A'A))
    else
        return trace(sqrtm(A * A'))
    end
end

function prox!{T <: Number}(A::AbstractMatrix{T}, p::NuclearNormPenalty{T}, s::T)
    svdecomp = svdfact!(A)
    soft_thresh!(svdecomp.S, s * p.λ)
    copy!(A, full(svdecomp))
end

function prox{T <: Number}(A::AbstractMatrix{T}, p::NuclearNormPenalty{T}, s::T)
    svdecomp = svdfact(A)
    soft_thresh!(svdecomp.S, s * p.λ)
    full(svdecomp)
end
