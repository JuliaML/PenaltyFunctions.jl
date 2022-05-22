"""
Penalties that are applied to the entire parameter array only
"""
abstract type ArrayPenalty <: Penalty end
name(p::ArrayPenalty) = replace(string(typeof(p)), "PenaltyFunctions." => "")

#------------------------------------------------------------------# abstract methods
value(p::ArrayPenalty, A::AbstractArray{<:Number}, λ::Number) = λ * value(p, A)


#----------------------------------------------------------------# NuclearNormPenalty
struct NuclearNormPenalty <: ArrayPenalty end
function value(p::NuclearNormPenalty, A::AbstractMatrix{<:Number})
    >(size(A)...) ? tr(sqrt(A'A)) : tr(sqrt(A * A'))
end
function prox!(p::NuclearNormPenalty, A::AbstractMatrix{<:Number}, λ::Number)
    svdecomp = svd!(A)
    soft_thresh!(svdecomp.S, λ)
    copyto!(A, Matrix(svdecomp))
end


#-----------------------------------------------------------------# GroupLassoPenalty
"Group Lasso Penalty.  Able to set the entire vector (group) to 0."
struct GroupLassoPenalty <: ArrayPenalty end
value(p::GroupLassoPenalty, A::AbstractArray{<:Number}) = norm(A)
function prox!(p::GroupLassoPenalty, A::AbstractArray{T}, λ::Number) where {T <: Number}
    denom = norm(A)
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
mutable struct MahalanobisPenalty{T <: Number, S <: AbstractArray{T,2}} <: ArrayPenalty
    C::S
    CtC::S
    CtC_Iλ::LU{T, Matrix{T}} # LU factorization of C'C + I/λ
    λ::T
end
function MahalanobisPenalty(C::AbstractMatrix{T}, λ::T = one(T)) where {T<:Number}
    MahalanobisPenalty(C, C'C, lu(C'C + I), λ)
end
function value(p::MahalanobisPenalty{T}, x::AbstractVector{T}) where {T <: Number}
    inv(T(2)) * T(sum(abs2, p.C * x))
end
function prox!(p::MahalanobisPenalty{T}, A::AbstractArray{T, 1}, λ::Number) where {T <: Number}
    if λ != p.λ
        p.λ = λ
        p.CtC_Iλ = lu(p.CtC + I / λ)
    end
    rmul!(A, one(T) / λ)
    ldiv!(p.CtC_Iλ, A) # overwrites result in A
end


#--------------------------------------------------------------------------------# scaled
struct ScaledArrayPenalty{T, P <: ArrayPenalty} <: ArrayPenalty
    penalty::P
    λ::T
end
scaled(p::ArrayPenalty, λ::Number) = (_scale_check(λ); ScaledArrayPenalty(p, λ))

Base.show(io::IO, sp::ScaledArrayPenalty) = print(io, "$(sp.λ) * ($(sp.penalty))")

value(p::ScaledArrayPenalty, θ::AbstractArray{<:Number}) = p.λ * value(p.penalty, θ)
prox!(p::ScaledArrayPenalty, θ::AbstractArray{<:Number}) = prox!(p.penalty, θ, p.λ)
