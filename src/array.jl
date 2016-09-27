# Penalties that evaluate on the entire array only
# TODO:
# - GeneralizedL1Penalty?
# - FusedL1Penalty

#------------------------------------------------------------------# abstract methods
value{T <: Number}(p::ArrayPenalty, A::AA{T}, s::T) = s * value(p, A)

prox!{T <: Number}(p::ArrayPenalty, A::AA{T}) = _prox!(p, A, p.λ)
prox!{T <: Number}(p::ArrayPenalty, A::AA{T}, s::T) = _prox!(p, A, s * p.λ)
prox{T <: Number}(p::ArrayPenalty, A::AA{T}) = prox!(p, deepcopy(A))
prox{T <: Number}(p::ArrayPenalty, A::AA{T}, s::T) = prox!(p, deepcopy(A), s)


#----------------------------------------------------------------# NuclearNormPenalty
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


#-----------------------------------------------------------------# GroupLassoPenalty
type GroupLassoPenalty{T <: Number} <: ArrayPenalty
    λ::T
end
GroupLassoPenalty(λ::Number = 0.1) = GroupLassoPenalty(λ)

value{T <: Number}(p::GroupLassoPenalty{T}, A::AA{T, 1}) = vecnorm(A)

function _prox!{T <: Number}(p::GroupLassoPenalty{T}, A::AA{T, 1}, s::T)
    scaling = s / vecnorm(A)
    for i in eachindex(A)
        @inbounds A[i] = soft_thresh(A[i], scaling * A[i])
    end
    A
end
