using Distributions

function cdf_transform(S)
    V = size(S)[1]

    ballot_budgets::Vector{Float64} = []
    transformed::Vector{Vector{Float64}} = []

    for v in 1:V
        ballot = S[v, :]
        prev_threshold = 0
        for p in sort(ballot)
            p == 0 && continue
            budget_interval = p - prev_threshold
            approved = (ballot .> prev_threshold)

            push!(ballot_budgets, budget_interval)
            push!(transformed, approved)

           prev_threshold = p
        end
    end

    @assert(0 ∉ ballot_budgets)
    return (Matrix(transpose(hcat(transformed...))), ballot_budgets)
end

function stochastic_bullets(S; voters=[])
    """
    A very simple min-max strategy where every voter picks a random approval threshold
    """
    V,C = size(S)
    length(voters) == 0 && begin voters = 1:V end
    S_trat = copy(S)
    for v in voters
        thresh = rand()
        S_trat[v, :] .= (S[v, :] .> thresh)
    end
    return S_trat
end

function frontrunner_bullets(S, W; voters=[])
    """
    Another very simple strategy where each voter clairvoyantly identifies
    the front-runners and min-maxes according to their average winner utility.
    """
    S_trat = copy(S)
    V,C = size(S_trat)
    for v in voters
        S_trat[v, :] .= zeros(C)
        adjusted_mean_frontrunner_utility = (0.75 + sum(S[v, W]))/(0.75 + length(W))
        S_trat[v, S[v, :] .≥ adjusted_mean_frontrunner_utility] .= 1
    end
    return S_trat
end

function topk_bullets(S, k; voters=[])
    """
    Each voter approves her k favorites
    """
    V,C = size(S)
    length(voters) == 0 && begin voters = 1:V end

    S_trat = copy(S)
    for v in voters
        S_trat[v, :] .= zeros(C)
        top_k_candidates = partialsortperm(S[v, :], k)
        S_trat[v, top_k_candidates] .= 1
    end
end