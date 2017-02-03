# PenaltyFunctions

| **Package Status** | **Package Evaluator** | **Build Status** |
|:------------------:|:---------------------:|:----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) | | [![Build Status](https://travis-ci.org/JuliaML/PenaltyFunctions.jl.svg?branch=master)](https://travis-ci.org/JuliaML/PenaltyFunctions.jl) [![Coverage Status](https://coveralls.io/repos/github/JuliaML/PenaltyFunctions.jl/badge.svg?branch=josh)](https://coveralls.io/github/JuliaML/PenaltyFunctions.jl?branch=josh)|


## Introduction
PenaltyFunctions is a collection of types for regularization in machine learning.

## Available Penalties



### Element-wise Penalties
*Penalties that apply to the parameter element-wise*


| **Value** | **Constraint Formulation** |
|:---------:|:--------------------------:|
| Univariate Parameter | Bivariate Parameter |
|![univariate](https://cloud.githubusercontent.com/assets/8075494/20890409/c778e6f2-bad4-11e6-9485-6886b84b741e.png) | TODO



- `NoPenalty()`
- `L1Penalty()`
- `L2Penalty()`
- `ElasticNetPenalty(α)`
- `SCADPenalty(a)`


<!-- ### Array Penalties
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
``` -->


## Example
TODO

## Documentation
TODO

## Installation
```julia
Pkg.clone("https://github.com/JuliaML/PenaltyFunctions.jl")
```

## License
This code is free to use under the terms of the MIT license.
