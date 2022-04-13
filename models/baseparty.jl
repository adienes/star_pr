using Distances
using Distributions
using Random

abstract type Party end
abstract type Election end

struct PartyElectorate
    #try to use "callable types" as well...
    hpuid::Int #unique identifier for following parameters

    k::Int #number of winners
    V::Int #number of voters
    C::Int #number of candidates

    M::Int #number of parties
    Q::Vector{Float64} #proportions of voters sampled from party
    L::Vector{Party}

    ω::Float64  #population proportion of strategic voters
    function PartyElectorate(k, V, C, partydata)
        hpuid = floor(Int, rand()*10^22)

        (M, Q, L) = partydata
        ω = rand()

        new(hpuid, k, V, C, M, Q, L, ω)
    end
end

function randparties(partysampler::Function; rangepartynum = 2:7)
    M = rand([j for j in rangepartynum])
    Q = rand(Dirichlet(ones(M)*8/M))
    L = [partysampler() for j in 1:M]
    return (M, Q, L)
end

function cast!(B::Election)
    E::PartyElectorate = B.E

    party_sampler = Categorical(E.Q)
    strat_sampler = Bernoulli(E.ω)

    voter_party::Vector{Int} = []
    voter_latent::Vector{Vector{Float64}} = []
    voter_strategy::Vector{Bool} = []
    for i=1:E.V
        party_affiliation = rand(party_sampler)

        push!(voter_party, party_affiliation)

        vl = getvoter(E.L[party_affiliation])
        vs = rand(strat_sampler)

        push!(voter_strategy, vs)
        push!(voter_latent, vl)
    end
    cands_latent = getcands(B; voters=voter_latent)

    S_incere = zeros(Float64, E.V, E.C)
    for j=1:E.C, i=1:E.V
        S_incere[i, j] = ξ(E.L[voter_party[i]], voter_latent[i], cands_latent[j])
    end

    B.latentvoters = voter_latent
    B.latentcands = cands_latent
    B.strategicbehavior = voter_strategy
    B.partyaffiliation = voter_party
    B.S = S_incere
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
    mins = mapslices(r -> minimum(r), S, dims=2)
    maxs = mapslices(r -> maximum(r), S, dims=2)
    temp = (S .- mins) ./ (maxs - mins)
    temp[isnan.(temp)] .= 0
    return temp
end