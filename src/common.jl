value(p::ElementPenalty, θ::Number, s::Number)       = s * value(p, θ)
value(p::ElementPenalty, θ::AbstractArray{<:Number})            = sum(x -> value(p, x), θ)
value(p::ElementPenalty, θ::AbstractArray{<:Number}, s::Number) = sum(x -> value(p, x, s), θ)
function value(p::ElementPenalty, θ::AbstractArray{T}, s::AbstractArray{S}) where {T <: Number, S <: Number}
    size(θ) == size(s) || error("lengths of parameter/weights do not match")
    result = zero(value(p, first(θ), first(s)))
    @inbounds for i in eachindex(θ, s)
        result += value(p, θ[i], s[i])
    end
    result
end

prox!(p::ProxableElementPenalty, θ::AbstractArray{<:Number}, s::Number) = map!(θj -> prox(p, θj, s), θ, θ)
function prox!(p::ProxableElementPenalty, θ::AbstractArray{<:Number}, s::AbstractArray{<:Number})
    @assert size(θ) == size(s)
    @inbounds for i in eachindex(θ, s)
        θ[i] = prox(p, θ[i], s[i])
    end
    θ
end
prox(p::ProxableElementPenalty, θ::AbstractArray{<:Number}, s::Number)       = prox!(p, copy(θ), s)
prox(p::ProxableElementPenalty, θ::AbstractArray{<:Number}, s::AbstractArray{<:Number}) = prox!(p, copy(θ), s)

deriv(p::ElementPenalty, θ::Number, s::Number) = s * deriv(p, θ)
grad(p::ElementPenalty, θ::AbstractArray{<:Number}) = grad!(similar(θ), p, θ)
function grad(p::ElementPenalty, θ::AbstractArray{T}, s::S) where {T<:Number, S<:Number}
    grad!(similar(θ, float(promote_type(T, S))), p, θ, s)
end
function grad(p::ElementPenalty, θ::AbstractArray{T}, s::AbstractArray{S}) where {T<:Number, S<:Number}
    grad!(similar(θ, float(promote_type(T, S))), p, θ, s)
end
function grad!(storage::AbstractArray{<:Number}, p::ElementPenalty, θ::AbstractArray{<:Number})
    map!(x -> deriv(p, x), storage, θ)
end
function grad!(storage::AbstractArray{<:Number}, p::ElementPenalty, θ::AbstractArray{<:Number}, s::Number)
    map!(x -> deriv(p, x, s), storage, θ)
end
function grad!(storage::AbstractArray{<:Number}, p::ElementPenalty, θ::AbstractArray{<:Number}, s::AbstractArray{<:Number})
    @assert size(storage) == size(θ) == size(s)
    @inbounds for j in eachindex(θ, s)
        storage[j] = deriv(p, θ[j], s[j])
    end
    storage
end

addgrad(∇j::Number, p::ElementPenalty, θj::Number) = ∇j + deriv(p, θj)
addgrad(∇j::Number, p::ElementPenalty, θj::Number, s::Number) = ∇j + s * deriv(p, θj)
function addgrad!(∇::AbstractArray{<:Number}, p::ElementPenalty, θ::AbstractArray{<:Number})
    @assert size(∇) == size(θ)
    @inbounds for j in eachindex(∇, θ)
        ∇[j] = addgrad(∇[j], p, θ[j])
    end
    ∇
end
function addgrad!(∇::AbstractArray{<:Number}, p::ElementPenalty, θ::AbstractArray{<:Number}, s::Number)
    @assert size(∇) == size(θ)
    @inbounds for j in eachindex(∇, θ)
        ∇[j] = addgrad(∇[j], p, θ[j], s)
    end
    ∇
end
function addgrad!(∇::AbstractArray{<:Number}, p::ElementPenalty, θ::AbstractArray{<:Number}, s::AbstractArray{<:Number})
    @assert size(∇) == size(θ) == size(s)
    @inbounds for j in eachindex(∇, θ, s)
        ∇[j] = addgrad(∇[j], p, θ[j], s[j])
    end
    ∇
end

value(p::ArrayPenalty, A::AbstractArray{<:Number}, λ::Number) = λ * value(p, A)

# --------------------
# AVAILABLE PENALTIES
# --------------------
include("penalties/elementwise.jl")
include("penalties/arraywise.jl")

# common functions
soft_thresh(x::Number, λ::Number) = sign(x) * max(zero(x), abs(x) - λ)

function soft_thresh!(x::AbstractArray{<:Number}, λ::Number)
    for i in eachindex(x)
        @inbounds x[i] = soft_thresh(x[i], λ)
    end
    x
end

# make penalties callable
for T in filter(!isabstracttype, union(subtypes(ElementPenalty), 
                                      subtypes(ProxableElementPenalty), 
                                      subtypes(ArrayPenalty)))
    @eval (pen::$T)(args...) = value(pen, args...)
end
