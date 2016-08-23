# Penalties

[![Build Status](https://travis-ci.org/JuliaML/Penalties.jl.svg?branch=master)](https://travis-ci.org/JuliaML/Penalties.jl)


# Warning
This package is in development and things may break/change

# Usage

```julia
using Penalties

p = L1Penalty(.1)
β = randn(5)
w = rand(5)
storage = zeros(5)

# Evaluate on Number
value(p, β[1])
deriv(p, β[1])
prox(p, β[1])

# Evaluate on Number with scaled λ
value(p, β[1], w[1])
deriv(p, β[1], w[1])
prox(p, β[1], w[1])

# Evaluate on array
value(p, β)
grad!(storage, p, β)
prox!(p, β)

# Evaluate on array with scaled λ
value(p, β, w[1])
grad!(storage, p, β, w[1])
prox!(p, β, w[1])

# Evaluate on array with element-wise scaled λ
value(p, β, w)
grad!(storage, p, β, w)
prox!(p, β, w)
```
