# PenaltyFunctions

| **Package Status** | **Package Evaluator** | **Build Status** |
|:------------------:|:---------------------:|:----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) | | [![Build Status](https://travis-ci.org/JuliaML/PenaltyFunctions.jl.svg?branch=master)](https://travis-ci.org/JuliaML/PenaltyFunctions.jl) [![Coverage Status](https://coveralls.io/repos/github/JuliaML/PenaltyFunctions.jl/badge.svg?branch=josh)](https://coveralls.io/github/JuliaML/PenaltyFunctions.jl?branch=josh)|


## Introduction
PenaltyFunctions is a collection of types for regularization in machine learning.

## Available Penalties



### Element Penalties
*Penalties that apply to the parameter element-wise*

An `ElementPenalty` has the form `sum(g, x)`

Penalty       | value on element
--------------|-----------------
`NoPenalty()` | `g(x) = 0`
`L1Penalty()` | `g(x) = abs(x)`
`L2Penalty()` | `g(x) = .5 * x ^ 2`
`ElasticNetPenalty(a)` | `g(x) = (1 - a) * abs(x) + a * .5 * x ^ 2`
`SCADPenalty(a)` | `L1Penalty that blends to constant`


```julia
using PenaltyFunctions
p = L1Penalty()
x = randn(5)
s = randn(5)
buffer = zeros(5)

# value
value(p, x[1])        # evaluate on element
value(p, x)           # evaluate on array
value(p, x[1], s[1])  # evaluate on element, scaled by scalar
value(p, x, s[1])     # evaluate on array, scaled by scalar
value(p, x, s)        # evaluate on array, element-wise scaling

# derivatives and gradients
deriv(p, x[1])        # derivative
deriv(p, x[1], s[1])  # scaled derivative
grad(p, x)            # gradient
grad(p, x, s[1])      # scaled gradient
grad(p, x, s)         # element-wise scaled gradient
grad!(buffer, p, x)       # overwrite buffer with gradient
grad!(buffer, p, x, s[1]) # overwrite buffer with scaled gradient
grad!(buffer, p, x, s)    # overwrite buffer with element-wise scaled gradient

# prox operator
prox(p, x[1], s[1]) # prox on element
prox(p, x, s[1])    # prox on array, scaled by scalar
prox(p, x, s)       # prox on array, element-wise scaling
prox!(p, x, s[1])   # overwrite x, scaled by scalar
prox!(p, x, s)      # overwrite x, element-wise scaling
```

### Array Penalties
*Penalties that need to be evaluated on the entire parameter*

Penalty                | value on array
-----------------------|-----------------
`NuclearNormPenalty()` | `sum of singular values of x`
`MahalanobisPenalty(C)`| `g(x) = x' * C' * C * x`
`GroupLassoPenalty()`  | `g(x) = vecnorm(x)`


## Installation
```julia
Pkg.clone("https://github.com/JuliaML/PenaltyFunctions.jl")
```

## License
This code is free to use under the terms of the MIT license.
