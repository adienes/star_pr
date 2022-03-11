    #Method of Equal Shares (aka rule X)
    #Can try Hare / Droop quota.
    #also completion is either
    #a) largest remainders
    #b) seq phragmen
    #c) seq dec quota
    #d) find lowest starting budget to be exhaustive
    #e) perturb input utilities to positive
    #f) multiply weights until can afford
    #g) find lowest price such that all winners purchased same price
    #       note: similar to d)

struct ElectionState
    V::Int
    C::Int
    q::Float64
    ballot_weights::Vector{Float64}
    remaining::Vector{Int}
    T::Matrix{Float64}
end

function allocated_score(S, k)
    return sequential_quota(S, k, allocate_utilitarian!)
end

function spent_score_scaling(S, k)
    return sequential_quota(S, k, spend_score_scaling!)
end

function single_transferable_vote(S, k)
    return sequential_quota(S, k, single_transfer_votes!)
end

function method_equal_shares(S, k)
    return sequential_quota(S, k, charge_equal_share!)
end

function spent_score_accumulation(S, k)
    return sequential_quota(S, k, spend_score_accumulating!)
end

function stratified_sortition(S, k)
    return sequential_quota(S, k, stratified_sortition!)
end

function sequential_quota(S::Array{Float64}, k, select_and_reweight!; complete=nothing)
    V,C = size(S)
    q = V/k
    ballot_weights::Vector{Float64} = ones(V)
    D = ElectionState(V, C, q, ballot_weights, Vector(1:C), copy(S))

    winner_set::Vector{Int} = []

    w = Inf
    while length(winner_set) < k && length(D.remaining) > 0 && !isnothing(w)
        w = select_and_reweight!(S, D)
        push!(winner_set, w)
        filter!(x -> x != w, D.remaining)
    end

    @assert(length(winner_set) ≤ k)
    return sort(winner_set)
end

function elimination_completion()
    if length(remaining_candidates) ≤ k - length(winner_set)
        winner_set = [winner_set; remaining_candidates]
        empty!(remaining_candidates)
    end
end
function single_transfer_votes!(S, D)
    """
    Quota can be: Hare, Droop, or Bottomup
    I should also try completion methods here like Shrinking Quota or Shrunken Quota
    I will implement it in a somewhat strange way where tied scores are treated like SAV,
    each taking an equal fraction of the voting power. If you want traditional STV, do not allow
    tied scores.
    """
    relative_contribution = 0
    surplus_factors = 0
    while true
        _S = @view S[:, D.remaining]

        #0-1 array of where voter's current support is
        preferred = (_S .== mapslices(x -> maximum(x), _S, dims=2))

        #what fraction of a voter's remaining ballot weight should be assigned
        relative_contribution = preferred ./ sum(preferred, dims=2)

        #current support for all remaining candidates
        surplus_factors = vec(D.ballot_weights'*relative_contribution) ./ D.q

        (any(surplus_factors .≥ 1) || length(D.remaining) == 1) && break

        lix = argmin(surplus_factors)
        l = D.remaining[lix]
        filter!(j -> j != l, D.remaining)
    end
    
    #There are some instantiations of STV where surplus votes can be transfered
    #recursively from winners to other winners. I do not allow that, but do
    #elect only one winner at a time.
    wix = argmax(surplus_factors)
    for v in 1:D.V
        D.ballot_weights[v] = max(0, D.ballot_weights[v] - relative_contribution[v, wix] / surplus_factors[wix])
    end

    return D.remaining[wix]
end

function charge_equal_share!(S, D)
    p_affordable = p -> vec(sum(min.(D.ballot_weights, S .* p), dims=1) .≥ D.q)[D.remaining]

    price_lowerbound = 0
    price_upperbound = 1 + 1 / minimum(S[S .> 0])

    p = (price_lowerbound + price_upperbound) / 2
    while price_upperbound - price_lowerbound > sqrt(eps())
        supported = p_affordable(p)

        if sum(supported) == 0
            price_lowerbound = p
        elseif sum(supported) ≥ 1
            price_upperbound = p
        end
        p = (price_lowerbound + price_upperbound) / 2
    end

    rho = price_upperbound
    supported = p_affordable(rho)
    wix = findfirst(x -> x > 0, supported)

    #Default to largest remainders completion
    isnothing(wix) && begin wix = argmax(vec(sum(D.ballot_weights .* S[:, D.remaining], dims=1))) end

    D.ballot_weights .= max.(0, D.ballot_weights - rho.*S[:, wix])
    return D.remaining[wix]
end

function allocate_utilitarian!(S, D)
    wix = argmax(vec(sum(D.ballot_weights .* S, dims=1))[D.remaining])
    w = D.remaining[wix]

    winner_scores = S[:, w]
    score_sorted = sortperm(winner_scores, rev=true)

    edge_voter = findfirst(r -> r ≥ D.q, cumsum(D.ballot_weights[score_sorted]))
    if isnothing(edge_voter)
        edge_score = 0
    else
        edge_score = winner_scores[score_sorted][edge_voter]
    end

    voters_above_edge = (winner_scores .> edge_score)
    D.ballot_weights[voters_above_edge] .== 0

    if edge_score > 0
        voters_on_edge = (winner_scores .== edge_score)
        weight_above_edge = sum(D.ballot_weights[voters_above_edge])
        weight_on_edge = sum(D.ballot_weights[voters_on_edge])
        surplus_factor = weight_on_edge / (D.q - weight_above_edge)
        D.ballot_weights[voters_on_edge] ./= surplus_factor
    else
        #Exhaustiveness---default completion is largest remainders
        #there may be other options though...  
    end

    return w
end

function spend_score_scaling!(S, D)
    weighted_scores = D.ballot_weights .* S
    supports = vec(sum(weighted_scores, dims=1))

    wix = argmax(supports[D.remaining])
    w = D.remaining[wix]

    surplus_factor = max(supports[w] / D.q, 1)
    D.ballot_weights .= max.(0, (D.ballot_weights - weighted_scores[:, w] / surplus_factor))

    return w
end

function spend_score_accumulating!(S, D)
    supports = vec(sum(D.T, dims=1))
    wix = argmax(supports[D.remaining])
    w = D.remaining[wix]

    surplus_factor = max(supports[w] / D.q, 1)
    score_spent = D.T[:, w] ./ surplus_factor

    for i in 1:D.V
        D.T[i, S[i, :] .≥ S[i, w]] .-= score_spent[i]

        running_min = 1
        for j in sortperm(S[i, :], rev=true)
            M = min(running_min, D.T[i, j])
            D.T[i, j] = M
            running_min = M
        end
    end

    return w
end

function stratified_sortition!(S, D)
    dictator = rand((1:D.V)[D.ballot_weights .!= 0])

    d_ballot = S[dictator, D.remaining]
    wix = rand(findall(d_ballot .== maximum(d_ballot)))
    w = D.remaining[wix]

    num_unexhausted = sum(D.ballot_weights)
    for i in 1:D.V
        rand() < (D.q / num_unexhausted) && begin D.ballot_weights[i] = 0 end
    end

    return w
end