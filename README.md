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

# most penalties have methods which operate on scalars or arrays
value(pen, β[1])
value(pen, β)

deriv(pen, β[1])
grad!(storage, pen, β)

prox!(pen, β[1])
prox!(pen, β)
```
