"""
Penalties that are applied to the entire parameter array only
"""
abstract ArrayPenalty <: Penalty

#------------------------------------------------------------------# abstract methods
value{T <: Number}(p::ArrayPenalty, A::AA{T}, λ::T) = λ * value(p, A)


#----------------------------------------------------------------# NuclearNormPenalty
immutable NuclearNormPenalty <: ArrayPenalty end
function value{T <: Number}(p::NuclearNormPenalty, A::AbstractMatrix{T})
    >(size(A)...) ? trace(sqrtm(A'A)) : trace(sqrtm(A * A'))
end
function prox!{T <: Number}(p::NuclearNormPenalty, A::AbstractMatrix{T}, λ::T)
    svdecomp = svdfact!(A)
    soft_thresh!(svdecomp.S, λ)
    copy!(A, full(svdecomp))
end


#-----------------------------------------------------------------# GroupLassoPenalty
"Group Lasso Penalty.  Able to set the entire vector (group) to 0."
immutable GroupLassoPenalty <: ArrayPenalty end
value{T <: Number}(p::GroupLassoPenalty, A::AA{T}) = vecnorm(A)
function prox!{T <: Number}(p::GroupLassoPenalty, A::AA{T}, λ::T)
    denom = vecnorm(A)
    if denom <= λ
        fill!(A, zero(T))
    else
        scaling = λ / denom
        for i in eachindex(A)
            @inbounds A[i] = (1 - scaling) * A[i]
        end
    end
    A
end


#-----------------------------------------------------------------# MahalanobisPenalty
"""
    MahalanobisPenalty(C)

Supports a Mahalanobis distance penalty (`xᵀCᵀCx` for a vector `x`).
"""
type MahalanobisPenalty{T <: Number} <: ArrayPenalty
    C::AA{T,2}
    CtC::AA{T,2}
    CtC_Iλ::Base.LinAlg.LU{T, Matrix{T}} # LU factorization of C'C + I/λ
    λ::T
end
function MahalanobisPenalty{T}(C::AbstractMatrix{T}, λ::T = one(T))
    MahalanobisPenalty(C, C'C, lufact(C'C + I), λ)
end
value{T}(p::MahalanobisPenalty{T}, x::AbstractVector{T}) = T(0.5) * sum(abs2, p.C * x)
function prox!{T <: Number}(p::MahalanobisPenalty{T}, A::AA{T, 1}, λ::T)
    if λ != p.λ
        p.λ = λ
        p.CtC_Iλ = lufact(p.CtC + I / λ)
    end
    scale!(A, 1 / λ)
    A_ldiv_B!(p.CtC_Iλ, A) # overwrites result in A
end
