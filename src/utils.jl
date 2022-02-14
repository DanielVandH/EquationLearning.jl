""" 
    scale_unit(x)

Scales the data in `x` such that `x ∈ [0, 1]`.
"""
function scale_unit(x)
    x_min = minimum(x)
    x_max = maximum(x)
    xx = @. (x - x_min) / (x_max - x_min)
    return xx
end