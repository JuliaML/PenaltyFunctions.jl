# Penalties

[![Build Status](https://travis-ci.org/joshday/Penalties.jl.svg?branch=master)](https://travis-ci.org/joshday/Penalties.jl)
[![codecov.io](http://codecov.io/github/joshday/Penalties.jl/coverage.svg?branch=master)](http://codecov.io/github/joshday/Penalties.jl?branch=master)


# Warning
This package is in development and things may break/change

# Usage

```julia
using Penalties
pen = L1Penalty(.1)
β = randn(5)
storage = zeros(5)

# element-wise penalties have methods which operate on scalars
value(pen, β[1])
deriv(pen, β[1])
prox!(pen, β[1])

# element-wise penalties also work with arbitrary arrays
value(pen, β)
grad!(storage, pen, β)
prox!(pen, β)

# Scaling parameters can also be used
# This provides element-wise tuning parameters λi = λ * scaling[i]
scaling = rand(5)

value(pen, β, scaling)
grad!(storage, pen, β, scaling)
prox!(pen, β, scaling)
```
