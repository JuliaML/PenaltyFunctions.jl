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
value{T <: Number}(p::GroupLassoPenalty, A::AbstractMatrix{T}) = vecnorm(A)
function prox!{T <: Number}(p::GroupLassoPenalty, A::AbstractMatrix{T}, λ::T)
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

#
# #-----------------------------------------------------------------# MahalanobisPenalty
# """
#     MahalanobisPenalty(λ, C)
#
# Supports a Mahalanobis distance penalty (`xᵀCᵀCx` for a vector `x`).
# """
# type MahalanobisPenalty{T <: Number} <: ArrayPenalty
#     λ::T
#     C::AA{T,2}
#     CtC::AA{T,2}
#     sλ::T
#     CtC_Isλ::Base.LinAlg.LU{T, Array{T,2}} # LU factorization of C'C + I/sλ
# end
# function MahalanobisPenalty{T}(λ::T, C::AA{T,2}, s::T=one(T))
#     MahalanobisPenalty(λ, C, C'C, s*λ, lufact(C'C + I/(λ*s)))
# end
# function MahalanobisPenalty{T}(C::AA{T,2}, s::T=one(T))
#     MahalanobisPenalty(one(T), C, C'C, s, lufact(C'C + I/s))
# end
#
# value{T <: Number}(p::MahalanobisPenalty{T}, x) = T(0.5) * p.λ * sumabs2(p.C * x)
#
# function _prox!{T <: Number}(p::MahalanobisPenalty{T}, A::AA{T, 1}, sλ::T)
#     if sλ != p.sλ
#         p.sλ = sλ
#         p.CtC_Isλ = lufact(p.CtC + I/sλ)
#     end
#
#     scale!(A, 1 / sλ)
#     A_ldiv_B!(p.CtC_Isλ, A) # overwrites result in A
# end
