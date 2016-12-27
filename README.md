# PenaltyFunctions




| **Package Status** | **Package Evaluator** | **Build Status** |
|:------------------:|:---------------------:|:----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) | | [![Build Status](https://travis-ci.org/JuliaML/PenaltyFunctions.jl.svg?branch=master)](https://travis-ci.org/JuliaML/PenaltyFunctions.jl) |


## Available Penalties



### Element-wise Penalties
*Penalties that apply to the parameter element-wise*


| **Value** | **Constraint Formulation** |
|:---------:|:--------------------------:|
| Univariate Parameter | Bivariate Parameter |
|![univariate](https://cloud.githubusercontent.com/assets/8075494/20890409/c778e6f2-bad4-11e6-9485-6886b84b741e.png) | TODO



- `NoPenalty()`
- `L1Penalty(λ)`
- `L2Penalty(λ)`
- `ElasticNetPenalty(λ, α)`
- `SCADPenalty(λ, a)`


### Array Penalties
*Penalties that need to be evaluated on the entire parameter*

- `NuclearNormPenalty(λ)`
- `MahalanobisPenalty(λ, C)`
- `GroupLassoPenalty(λ)`

```julia
Θ = randn(10, 5)

p = NuclearNormPenalty(.1)

value(p, Θ)
prox(p, Θ)
prox!(p, Θ)
```


## Example
```julia
using PenaltyFunctions

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

## Documentation
TODO

## Installation
```julia
Pkg.clone("https://github.com/JuliaML/PenaltyFunctions.jl")
```

## License
This code is free to use under the terms of the MIT license.
