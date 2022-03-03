using Distances
using Distributions

#=
mallows mixture model
polya urn model (includes ic, iac)
linear / concave / convex ranking utility functions
=#

function uniform_ic(V, C)
    return rand(V, C)
end

function cube_polar_uniform(V, C)
    candidate_positions = [rand(3) for _ in 1:C]
    voter_positions = [rand(3) for _ in 1:V]
    
    S = zeros(V, C)
    for c in 1:C
        cpos = candidate_positions[c]
        S[:, c] .= [max(0, 1-1.6*cosine_dist(vpos, cpos)) for vpos in voter_positions]
    end

    return S
end

function square_euclid_uniform(V, C)
    candidate_positions = [rand(2) for _ in 1:C]
    voter_positions = [rand(2) for _ in 1:V]

    S = zeros(V,C)
    for c in 1:C
        cpos = candidate_positions[c]
        S[:, c] .= [max(0, 1 - 1.6*euclidean(vpos, cpos)) for vpos in voter_positions]
    end

    return S
end

function telos_linf_unichlet(V, C)
    alpha = rand(3).*1.6 .+1
    d = Dirichlet(alpha)

    candidate_positions = [rand(d) for _ in 1:C]
    voter_positions = [rand(d) for _ in 1:V]

    S = zeros(V, C)
    for c in 1:C
        cpos = candidate_positions[c]
        S[:, c] .= [max(0, 1-1.8*chebyshev(vpos, cpos)) for vpos in voter_positions]
    end

    return S
end