"""
Penalties that are applied to the entire parameter array only
"""
abstract type ArrayPenalty <: Penalty end
name(p::ArrayPenalty) = replace(string(typeof(p)), "PenaltyFunctions.", "")

#------------------------------------------------------------------# abstract methods
value(p::ArrayPenalty, A::AA{<:Number}, λ::Number) = λ * value(p, A)


#----------------------------------------------------------------# NuclearNormPenalty
immutable NuclearNormPenalty <: ArrayPenalty end
function value(p::NuclearNormPenalty, A::AbstractMatrix{<:Number})
    >(size(A)...) ? trace(sqrtm(A'A)) : trace(sqrtm(A * A'))
end
function prox!(p::NuclearNormPenalty, A::AbstractMatrix{<:Number}, λ::Number)
    svdecomp = svdfact!(A)
    soft_thresh!(svdecomp.S, λ)
    copy!(A, full(svdecomp))
end


#-----------------------------------------------------------------# GroupLassoPenalty
"Group Lasso Penalty.  Able to set the entire vector (group) to 0."
immutable GroupLassoPenalty <: ArrayPenalty end
value(p::GroupLassoPenalty, A::AA{<:Number}) = vecnorm(A)
function prox!{T <: Number}(p::GroupLassoPenalty, A::AA{T}, λ::Number)
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
value{T}(p::MahalanobisPenalty{T}, x::AbstractVector{T}) = float(T)(0.5) * T(sum(abs2, p.C * x))
function prox!{T <: Number}(p::MahalanobisPenalty{T}, A::AA{T, 1}, λ::Number)
    if λ != p.λ
        p.λ = λ
        p.CtC_Iλ = lufact(p.CtC + I / λ)
    end
    scale!(A, one(T) / λ)
    A_ldiv_B!(p.CtC_Iλ, A) # overwrites result in A
end


#--------------------------------------------------------------------------------# scaled
immutable ScaledArrayPenalty{T, P <: ArrayPenalty} <: ArrayPenalty
    penalty::P
    λ::T
end
scaled(p::ArrayPenalty, λ::Number) = (_scale_check(λ); ScaledArrayPenalty(p, λ))

Base.show(io::IO, sp::ScaledArrayPenalty) = print(io, "$(sp.λ) * ($(sp.penalty))")

value(p::ScaledArrayPenalty, θ::AA{<:Number}) = p.λ * value(p.penalty, θ)
prox!(p::ScaledArrayPenalty, θ::AA{<:Number}) = prox!(p.penalty, θ, p.λ)
