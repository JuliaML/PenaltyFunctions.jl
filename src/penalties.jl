# penalty functions
include("elementpenalty.jl")
include("arraypenalty.jl")

# common functions
soft_thresh(x::Number, λ::Number) = sign(x) * max(zero(x), abs(x) - λ)

function soft_thresh!(x::AbstractArray{<:Number}, λ::Number)
    for i in eachindex(x)
        @inbounds x[i] = soft_thresh(x[i], λ)
    end
    x
end

# Make Penalties Callable
for T in filter(!isabstracttype, union(subtypes(ElementPenalty), 
                                      subtypes(ProxableElementPenalty), 
                                      subtypes(ArrayPenalty)))
    @eval (pen::$T)(args...) = value(pen, args...)
end
