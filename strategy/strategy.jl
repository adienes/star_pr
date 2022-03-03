using Distributions

function normalize_scores(S)
    nzrows = findall(i -> !i, iszero.(eachrow(S)))
    nzcols = findall(i -> !i, iszero.(eachcol(S)))
    S = S[nzrows, nzcols]

    mins = mapslices(r -> minimum(r), S, dims=2)
    maxs = mapslices(r -> maximum(r), S, dims=2)

    return (S .- mins) ./ (maxs - mins)
end

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

function in_sincere_mix(S, k; strategic_percentage = 0.5, strategic_distribution=[0.28,0.36,0.36])
    V = size(S)[1]

    is_strat = Bernoulli(strategic_percentage)
    which_strat = Multinomial(1, strategic_distribution)
    sub = []
    frb = []
    tkb = []
    for v in 1:V
        !(rand(is_strat)) && continue
        push!([sub,frb,tkb][findfirst(Bool.rand(which_strat))], v)
    end

    S_trat = stochastic_bullets(S; voters=sub)
    S_trat = frontrunner_bullets(S_trat, k; voters=frb)
    S_trat = topk_bullets(S_trat, k; voters=tkb)
    return S_trat
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

function frontrunner_bullets(S, k; voters=[])
    """
    Another very simple strategy where each voter identifies the front-runners
    and min-maxes accordingly.
    """
    V,C = size(S)
    length(voters) == 0 && begin voters = 1:V end

    frontrunners = partialsortperm(sum(S, dims=1), k)

    S_trat = copy(S)
    for v in voters
        S_trat[v, :] .= zeros(C)
        favored_frontrunner_utility = maximum(S[v, frontrunners])
        favored_frontrunner_utility == 0 && begin favored_frontrunner_utility = eps() end
        
        S_trat[v, S[v, :] .≥ favored_frontrunner_utility] .= 1
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