#=
Distributions of (EVE = Expectation, Variance, Extremes)

* (EVE) support; over candidates, given balanced weight vector for phragmms objective

* stable priceability

=#

include("../strategy/strategy.jl")
using LinearAlgebra
using Statistics
using Distances

using JuMP
#empirically the fastest...
#compared to GLPK, COSMO, Cbc, Ipopt, Ipopt
using Clp
#using COSMO
using Ipopt

function get_metrics(S, W, V, C, k)
    S ./= maximum(S)
    #ALL THESE METRICS SHOULD BE NORMALIZED BY THE BEHAVIOR OF A RANDOM SELECTION
    #this will make it much more domain robust

    m = Dict()

    #Utilitarian metrics
    m["Expected Linear Utility"] = mean(sum(S[:, W], dims=2))
    # m["Variance Linear Utility"] = var(sum(S[:, W], dims=2))
    # m["Expected Log Utility"] = mean(log1p.(sum(S[:, W], dims=2)))
    # m["Variance Log Utility"] = var(sum(S[:, W], dims=2))
    m["Expected Max Utility"] = mean(maximum(S[:, W], dims=2))
    # m["Variance Max Utility"] = var(maximum(S[:, W], dims=2))

    # m["Expected Winner Polarization"] = mean(mapslices(c -> var(c), S[:, W], dims=1))
    # m["Variance Winner Polarization"] = var(mapslices(c -> var(c), S[:, W], dims=1))
    # m["Least Polarized Winner"] = minimum(mapslices(c -> var(c), S[:, W], dims=1))
    # m["Most Polarized Winner"] = maximum(mapslices(c -> var(c), S[:, W], dims=1))
    
    # m["Most Blocking Loser Capture"] = maximum(sum(S .* (S .> sum(S[:, W], dims=2)), dims=1)) / V
    # m["Average Utility Gain Extra Winner"] = maximum(sum(clamp.((S .- sum(S[:, W], dims=2)), 0, 1), dims=1)) / V
    # m["Effective Rank Reduction Ratio"] = effective_rank(S[:, W]) / effective_rank(S)
    # m["Sortition Efficiency"] = sortition_efficiency(S, W, k)

    #also Tukey depth min/max/mean/var would be cool to look at per winner

    #m["Maximin Support"] = maximin_support(stochastic_bullets(S), W, k)
    #m["Stable Price"] = stable_price(stochastic_bullets(S), W, k)
    #m["Load Dispersion"] = load_dispersion(stochastic_bullets(S), W, k)

    #These metrics have a geometric interpretation. Also they are in the model
    #like MAV, where a vote for all candidates means specifically ALL candidates, not a subset
    # w_oz = [c ∈ W ? 1 : 0 for c in 1:C]
    # m["Expected Cosine Distance"] = mean(mapslices(b -> cosine_dist(b, w_oz), S, dims=2))
    # m["Variance Cosine Distance"] = var(mapslices(b -> cosine_dist(b, w_oz), S, dims=2))
    # m["Expected Euclidean Distance"] = mean(mapslices(b -> euclidean(b, w_oz), S, dims=2))
    # m["Variance Euclidean Distance"] = var(mapslices(b -> euclidean(b, w_oz), S, dims=2))
    return m
end


function committee_is_preferred(A, S, T)
    return (schulze_committee_preference(A, S, T) - schulze_committee_preference(A, T, S)) > 0
end

function maximin_support(A, W, k; initial_budgets = ones(size(A)[1]))
    #=
    WARNING: do not expect reasonable results if A is not an approval matrix.
    =#
    V = size(A)[1]
    wa = A[:, W]

    model = Model(Clp.Optimizer)
    set_optimizer_attribute(model, "LogLevel", 0)
    
    @variable(model, 0 ≤ x[i = 1:V, j = 1:k] ≤ wa[i, j])
    @variable(model, s ≥ 0)

    @constraint(model, sum(x, dims=2) .≤ initial_budgets)
    @constraint(model, sum(x, dims=1) .≥ s)

    @objective(model, Max, s)

    optimize!(model)
    rv = objective_value(model) * k / V
    return rv
end

function stable_price(A, W, k; initial_budgets = ones(size(A)[1]))
    #=
    WARNING: do not expect reasonable results if A is not an approval matrix.
    =#
    V,C = size(A)
    L = [i for i in 1:C if i ∉ W]

    wa = A[:, W]
    wl = A[:, L]

    model = Model(Clp.Optimizer)
    set_optimizer_attribute(model, "LogLevel", 0)

    @variable(model, 0 ≤ x[i = 1:V, j = 1:k] ≤ wa[i, j]) #C1, C4
    @variable(model, 0 ≤ p)
    @variable(model, b[i = 1:V]) #maximum variables
    r = 1 .- x*ones(k)

    @constraint(model, sum(x, dims=2) .≤ initial_budgets) #C2
    @constraint(model, sum(x, dims=1) .== p) #C3

    @constraint(model, b .≥ x)
    @constraint(model, b .≥ r)
    @constraint(model, adjoint(b'*wl) .≤ p) #S5

    @objective(model, Max, p)
    optimize!(model)

    rv = objective_value(model) * k / V
    if termination_status(model) == MOI.INFEASIBLE
        return 0
    else
        return rv
    end
end

function load_dispersion(A, W, k; initial_budgets = ones(size(A)[1]))
    #=
    WARNING: do not expect reasonable results if A is not an approval matrix.
    =#
    V = size(A)[1]
    wa = A[:, W]

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    
    @variable(model, 0 ≤ x[i = 1:V, j = 1:k] ≤ initial_budgets.*wa[i, j])
    @constraint(model, sum(x, dims=1) .== 1)

    loads = sum(x, dims=2)
    @objective(model, Min, dot(loads, loads))

    optimize!(model)
    rv = objective_value(model)
    return rv
end

function schulze_committee_preference(A, s, t)
    #=
    WARNING: This looks only at pairwise preference information. It does not care about relative magnitudes.
    =#
    sort(s) == sort(t) && return 0

    V, m = size(A)[1], length(s)
    s_approvals = (maximum(A[:, filter(j -> j ∉ s, t)], dims=2) .< A)[:, s]

    model = Model(Clp.Optimizer)
    set_optimizer_attribute(model, "LogLevel", 0)
    
    @variable(model, 0 ≤ x[i = 1:V, j = 1:m] ≤ s_approvals[i, j])
    @variable(model, t ≥ 0)

    @constraint(model, sum(x, dims=2) .≤ 1)
    @constraint(model, sum(x, dims=1) .≥ t)

    @objective(model, Max, t)

    optimize!(model)
    return objective_value(model)
end

function effective_rank(A)
    singular_values = svd(A).S
    p = singular_values ./ sum(singular_values)
    return exp(-sum(p'*log.(p)))
end

function sortition_efficiency(A, W, k)
    V,C = size(A)
    
    upstream = mapslices(x -> partialsortperm(x, 1:k; rev=true), A', dims=1)
    kde = zeros((C,V))

    for v in 1:V
        kde[upstream[:, v], v] .= 1.0
    end

    kde = vec(sum(kde, dims=2)) ./ V
    w_oz = [c ∈ W ? 1 : 0 for c in 1:C]

    return (kde'*w_oz)/k
end