using Distances
using Distributions
using Random

abstract type Party end
abstract type Election end

struct PartyElectorate
    hpuid::Int #unique identifier for following parameters

    k::Int #number of winners
    V::Int #number of voters
    C::Int #number of candidates

    M::Int #number of parties
    Q::Vector{Float64} #proportions of voters sampled from party
    L::Vector{Party}

    r::Float64  #population proportion of strategic voters
    function PartyElectorate(k, V, C, partydata)
        hpuid = floor(Int, rand()*10^22)

        (M, Q, L) = partydata
        r = sum(*(x...) for x in zip(Q, [P.ω for P in L]))

        new(hpuid, k, V, C, M, Q, L, r)
    end
end

function cast!(B::Election)
    E::PartyElectorate = B.E

    party_sampler = Categorical(E.Q)

    voter_party::Vector{Int} = []
    voter_strategy::Vector{Bool} = []
    voter_latent::Vector{Vector{Float64}} = []
    for i=1:E.V
        party_affiliation = rand(party_sampler)

        push!(voter_party, party_affiliation)

        (vl, vs) = getvoter(E.L[party_affiliation])
        push!(voter_strategy, vs)
        push!(voter_latent, vl)
    end
    cands_latent = getcands(B)

    S_incere = zeros(Float64, E.V, E.C)
    for j=1:E.C, i=1:E.V
        S_incere[i, j] = ξ(E.L[voter_party[i]], voter_latent[i], cands_latent[j])
    end
    S_incere = normalizescores(S_incere)

    B.latentvoters = voter_latent
    B.latentcands = cands_latent
    B.strategicbehavior = voter_strategy
    B.partyaffiliation = voter_party
    B.S = S_incere
end

function randparties(partysampler::Function; rangepartynum = 1:7)
    M = rand([j for j in rangepartynum])
    Q = rand(Dirichlet(ones(M)))
    L = [partysampler() for j in 1:M]
    return (M, Q, L)
end

#Party-based models above
###########################
#Idiosyncratic models below

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

function normalizescores(S)
    nzrows = findall(i -> !i, iszero.(eachrow(S)))
    nzcols = findall(i -> !i, iszero.(eachcol(S)))
    S = S[nzrows, nzcols]

    mins = mapslices(r -> minimum(r), S, dims=2)
    maxs = mapslices(r -> maximum(r), S, dims=2)

    return (S .- mins) ./ (maxs - mins)
end
