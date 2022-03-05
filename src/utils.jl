#####################################################################
## Script description: utils.jl 
##
## This script contains certain functions to help with other computations.
##
## The following functions are defined:
##  - scale_unit: Scales data to be in a unit interval.
##  - searchsortednearest: Given a sorted array and a point, find the
##      closest element in the array to the point.
##  - data_thresholder: Gives indices corresponding to reliable data 
##      points.
##  - compute_ribbon_features: For a given vector x, computes the mean values and lower/upper bounds for a ribbon plot.
##
#####################################################################

""" 
    scale_unit(x)

Scales the data in `x` such that `x ∈ [0, 1]`.

# Arguments 
- `x`: The data to be scaled. 

# Outputs 
- `xx`: The scaled data such that `xx ∈ [0, 1]`.
"""
function scale_unit(x)
    x_min = minimum(x)
    x_max = maximum(x)
    xx = @. (x - x_min) / (x_max - x_min)
    return xx
end

"""
    searchsortednearest(a, x; <keyword arguments>)

Finds the index in `a` that has the smallest `distance` to `x`. Ties go to the smallest index.

*Source*: Taken from https://github.com/joshday/SearchSortedNearest.jl/blob/main/src/SearchSortedNearest.jl.

# Arguments 
- `a`: A sorted collection.
- `x`: A value which is used for comparing the values of `a` to for finding the closest value.

# Keyword Arguments 
- `by = identity`: The order in which `a` is sorted. If `a` is sorted, this is just `identity`.
- `lt = isless`: The metric used for sorting values in `a`, typically `lt = isless` which uses `<`.
- `rev = false`: Whether to reverse the array `a`.
- `distance = (a, b) -> abs(a - b)`: The distance metric used for assessing which value is closest.

# Outputs
- `i`: The index such that `distance(a[i], a)` is minimised.
"""
function searchsortednearest(a, x; by = identity, lt = isless, rev = false, distance = (a, b) -> abs(a - b))
    i = searchsortedfirst(a, x; by, lt, rev)
    if i == 1
    elseif i > length(a)
        i = length(a)
    elseif a[i] == x
    else
        i = lt(distance(by(a[i]), by(x)), distance(by(a[i-1]), by(x))) ? i : i - 1
    end
    return i
end

"""
    data_thresholder(f, fₜ, τ)

Thresholds the data given by `f` with temporal derivatives `fₜ` based on a threshold tolerance `τ`. The returned values 
in `inIdx` give the indices in `f` (and `fₜ`) that should be kept. 

# Arguments 
- `f`: Computed values of `f(x, t)`.
- `fₜ`: Computed values of `fₜ(x, t)`.
- `τ`: A tuple of the form `(τ₁, τ₂)` which gives the tolerance `τ₁` for thresholding `f` and `τ₂` for thresholding `fₜ`.

# Outputs 
- `inIdx`: Indices in `f` (and `fₜ`) that should be kept.

# Extended help 
The threshold conditions are:

    1. `min(|f|)τ₁ ≤ |f| ≤ max(|f|)(1-τ₁)`.
    2. `min(|fₜ|)τ₂ ≤ |fₜ| ≤ max(|fₜ|)(1-τ₂2)`.
    3. `f ≥ 0.0`.
"""
function data_thresholder(f, fₜ, τ)
    absf = abs.(f)
    absfₜ = abs.(fₜ)
    cond1 = minimum(absf)τ[1] .≤ absf .≤ maximum(absf) * (1 - τ[1])
    cond2 = minimum(absfₜ)τ[2] .≤ absfₜ .≤ maximum(absfₜ) * (1 - τ[2])
    cond3 = f .≥ 0.0
    inIdx = findall(cond1 .& cond2 .& cond3)
    return inIdx
end

"""
    compute_ribbon_features(x; level = 0.05)

Computes features for a confidence interval plot for some data `x`.

# Arguments 
- `x`: The data to use for plotting.

# Keyword Arguments 
- `level = 0.05`: The significance level for computing the credible intervals for the parameter values. 

# Outputs 
- `x_mean`: The mean of each row of `x`.
- `x_lower`: The `100(`level/2`)%` quantile for each row of `x`.
- `x_upper`: The `100(`1-level/2`)%` quantile for each row of `x`.
"""
function compute_ribbon_features(x; level = 0.05)
    x_mean = [mean(r) for r in eachrow(x)]
    x_lower = [quantile(r, level / 2) for r in eachrow(x)]
    x_upper = [quantile(r, 1 - level / 2) for r in eachrow(x)]
    return x_mean, x_lower, x_upper
end
