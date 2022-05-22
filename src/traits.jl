"""
Baseclass for all penalties.
"""
abstract type Penalty end

"""
Penalties that are applied element-wise.
"""
abstract type ElementPenalty <: Penalty end
abstract type ProxableElementPenalty <: ElementPenalty end

"""
Penalties that are applied to the entire parameter array.
"""
abstract type ArrayPenalty <: Penalty end