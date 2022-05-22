# PenaltyFunctions

_PenaltyFunctions.jl is a Julia package that provides generic
implementations for a diverse set of penalty functions that are
commonly used for regularization purposes in Machine Learning._

[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Build Status](https://img.shields.io/github/workflow/status/JuliaML/PenaltyFunctions.jl/CI?style=flat-square)](https://github.com/JuliaML/PenaltyFunctions.jl/actions)
[![Coverage Status](https://img.shields.io/codecov/c/github/JuliaML/PenaltyFunctions.jl?style=flat-square)](https://codecov.io/gh/JuliaML/PenaltyFunctions.jl)

## Introduction

Many popular models in Machine Learning are parameterized by a
set of real-valued coefficients `θ` (theta), which is usually
stored in the form of an array. If our data set has `k` features,
then `θ` would typically be a vector of `k` or `k+1` numeric
elements. Each individual feature `x_i` of our data set is
assigned a corresponding coefficient `θ_i`, which is used to
quantify the feature's influence on the prediction. The concrete
values for the coefficient vector are learned by an optimization
algorithm, which tries to select the "best" set of coefficients
for the given data and model. Without any restriction on their
values the optimization algorithm is free to choose the
coefficients freely, which may result in overly complex
prediction functions. This freedom is known to cause the
optimization algorithm to overfit to the noise in the training
data. This is where penalties come in!

A penalty is a function of the coefficients and only the
coefficients. It associates the given set of coefficients with a
cost without any regard for their influence on the predictive
power of the prediction function. This cost is then is added to
the overall cost of the prediction function. This way the
optimization algorithm is encouraged to choose "simpler"
coefficients. What exactly "simpler" means depends on the chosen
penalty. In general terms: penalties help to reduce the
possibility of overfitting.

## Available Penalties

This package implements a number of carefully crafted penalty
functions, as well as an API to query their properties (e.g.
convexity). Furthermore, we expose methods to compute their
values and derivatives for a single value, coefficient vectors,
and even arrays of arbitrary dimensionality. The provided penalty
functions fall into one of two main families, namely **Element
Penalties** and **Array Penalties**.

### Element Penalties

The first family of penalty functions contains all those that
apply to to the individual elements of `θ` element-wise. The
resulting cost of a coefficient array is then the sum of the
element-wise results.

**Univariate Parameter** | **Bivariate Parameter**
:-----------------------:|:-----------------------:
![univariate_elem](https://rawgithub.com/JuliaML/FileStorage/master/PenaltyFunctions/univariate.svg) | ![bivariate_elem](https://rawgithub.com/JuliaML/FileStorage/master/PenaltyFunctions/bivariate.svg)
The cost-values of various penalties as a function of a single coefficient | Cross sections of the cost-surfaces. This time for two coefficients

Every penalty that is of this family is subtype of
`ElementPenalty`. From an implementation perspective these
penalties are defined using the element-wise functions. The
following table lists the implemented types and their
definitions.

Penalty       | value on element
--------------|-----------------
`NoPenalty()` | `g(θ) = 0`
`L1Penalty()` | `g(θ) = abs(θ)`
`L2Penalty()` | `g(θ) = 0.5 * θ ^ 2`
`ElasticNetPenalty(α = 0.5)` | `g(θ) = (1 - α) * abs(θ) + α * .5 * θ ^ 2`
`SCADPenalty(a = 3.7, γ = 1.0)` | `L1Penalty that blends to constant`
`MCPPenalty(γ = 2.0)` | `g(θ) = abs(θ) < γ ? abs(θ) - θ ^ 2 / 2γ : γ / 2`
`LogPenalty(η = 1.0)` | `g(θ) = log(1 + η * abs(θ))`

The total cost for an array of coefficients is then defined as
`sum(g, θ)`.

```julia
using PenaltyFunctions
p = L1Penalty()
x = randn(5)
s = randn(5)
buffer = zeros(5)

# value
value(p, x[1])        # evaluate on element
value(p, x)           # evaluate on array
value.(p, x)          # broadcast is supported as well
value(p, x[1], s[1])  # evaluate on element, scaled by scalar
value(p, x, s[1])     # evaluate on array, scaled by scalar
value(p, x, s)        # evaluate on array, element-wise scaling

# value via calling the Penalty object
p = L1Penalty()
p([1,2,3])

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

The second family of penalty functions contains all those that
that need to be evaluated on the entire coefficient array `θ` at
once. Every penalty that belongs to this family is subtype of
`ArrayPenalty`. The following table outlines the implemented
types and their definitions.

Penalty                | value on array
-----------------------|-----------------
`NuclearNormPenalty()` | `sum of singular values of x`
`MahalanobisPenalty(C)`| `g(x) = x' * C' * C * x`
`GroupLassoPenalty()`  | `g(x) = vecnorm(x)`

## Installation

The package is registered in `METADATA.jl`.

```julia
Pkg.add("PenaltyFunctions")
```

## License

This code is free to use under the terms of the MIT "Expat" license.
