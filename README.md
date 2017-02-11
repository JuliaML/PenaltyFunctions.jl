# PenaltyFunctions

| **Package Status** | **Package Evaluator** | **Build Status** |
|:------------------:|:---------------------:|:----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) | | [![Build Status](https://travis-ci.org/JuliaML/PenaltyFunctions.jl.svg?branch=master)](https://travis-ci.org/JuliaML/PenaltyFunctions.jl) [![Coverage Status](https://coveralls.io/repos/github/JuliaML/PenaltyFunctions.jl/badge.svg?branch=josh)](https://coveralls.io/github/JuliaML/PenaltyFunctions.jl?branch=josh)|


## Introduction
PenaltyFunctions is a collection of types for regularization in machine learning.

## Available Penalties



### Element Penalties
*Penalties that apply to the parameter element-wise*

![](https://cloud.githubusercontent.com/assets/8075494/22839303/d5269fe6-ef96-11e6-8c27-29b52d5e4499.png)

An `ElementPenalty` has the form `sum(g, θ)`

Penalty       | value on element
--------------|-----------------
`NoPenalty()` | `g(θ) = 0`
`L1Penalty()` | `g(θ) = abs(θ)`
`L2Penalty()` | `g(θ) = .5 * θ ^ 2`
`ElasticNetPenalty(α = 0.5)` | `g(θ) = (1 - α) * abs(θ) + α * .5 * θ ^ 2`
`SCADPenalty(a = 3.7, γ = 1.0)` | `L1Penalty that blends to constant`
`MCPPenalty(γ = 2.0)` | `g(θ) = abs(θ) < γ ? abs(θ) - θ ^ 2 / 2γ : γ / 2`
`LogPenalty(η = 1.0)` | `g(θ) = log(1 + η * abs(θ))`


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
