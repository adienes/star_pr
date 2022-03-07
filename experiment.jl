include("./metrics/metrics.jl")
include("./models/models.jl")
include("./rules/quota_spending.jl")
include("./rules/other_rules.jl")

using DataFrames
using Distributions
using LinearAlgebra
using Random

struct SpatialParty{N}
    μ::Vector{Float64} #center of multivariate gaussian
    σ::Vector{Float64} #stdev along each axis

    λ::Float64 #natural disposition
    ω::Float64 #proportion of strategic voters

    function SpatialParty{N}() where {N}
        μ = (rand(N) + rand(N)) .- 1 #slight centrist bias on [-1, +1]
        σ = rand(N)*0.28 #tip-to-tip about 20% utility

        λ = sqrt(rand())
        ω = rand()
        new{N}(μ, σ, λ, ω)
    end
end

struct MallowsParty
    α::Vector{Int} #reference ranking
    ϕ::Float64 #dispersion probability

    λ::Float64 #natural disposition
    ω::Float64 #proportion of strategic voters

    function MallowsParty()
        #these parameters imply each voter ranks around 10 candidates
        #parties on average overlap by ~3 candidates
        α = shuffle(unique(rand(1:30, 10)))
        ϕ = (rand() + rand() + rand())/3
        
        λ = sqrt(rand())
        ω = rand()
        new(α, ϕ, λ, ω)
    end
end

function getvoter(P::SpatialParty)
    return (rand.([Normal(x...) for x in zip(P.μ, P.σ)]), rand(Bernoulli(P.ω)))
end

function getvoter(P::MallowsParty)
    r::Vector{Int} = []
    for (i,c) in enumerate(P.α)
        denom = sum([P.ϕ^i for i in 0:(i-1)])
        pos = rand(Categorical([P.ϕ^(i - j)/denom for j in 1:i]))
        insert!(r, pos, c)
    end
    append!(r, shuffle(setdiff(1:30, r)))

    return (r, rand(Bernoulli(P.ω)))
end

struct Electorate
    hpuid::Int #unique identifier for following parameters

    k::Int #number of winners
    V::Int #number of voters
    C::Int #number of candidates

    M::Int #number of parties
    Q::Vector{Float64} #proportions of voters sampled from party
    L::Vector{Union{SpatialParty, MallowsParty}}

    r::Float64  #population proportion of strategic voters
    function Electorate(k, V, C; Party=SpatialParty{2})
        hpuid = floor(Int, rand()*10^22)
        M = rand(1:7)
        L = [Party() for j in 1:M]
        Q = rand(Dirichlet(ones(M)))
        r = sum(*(x...) for x in zip(Q, [P.ω for P in L]))

        new(hpuid, k, V, C, M, Q, L, r)
    end
end

function getcands(E::Electorate)
    if typeof(E.L[1]) <: MallowsParty
        return [j for j in 1:30]
    elseif typeof(E.L[1]) <: SpatialParty
        N = typeof(E.L[1]).parameters[1]
        return [(rand(N) + rand(N)) .- 1 for j in 1:E.C]
    else
        return nothing
    end
end

function ξ(P::SpatialParty, v::Vector{Float64}, c::Vector{Float64})
    return max(1 - P.λ*2*norm(c-v), 0)
end

function ξ(P::MallowsParty, v::Vector{Int}, c::Int)
    rank = findfirst(j -> j == c, v)
    isnothing(rank) && return 0
    return P.λ^(rank-1)
end

function sampleutilities(E::Electorate)    
    #ξ takes a set of latent voters to utilities

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

    cands_latent = getcands(E)

    S_incere = zeros(Float64, E.V, E.C)

    for j=1:E.C, i=1:E.V
        S_incere[i, j] = ξ(E.L[voter_party[i]], voter_latent[i], cands_latent[j])
    end

    return (S_incere, voter_strategy)
end


Ec = Electorate(5, 100, 15)

display(sampleutilities(Ec)[1])
@assert(false)

function main()
    
    metric_keys = []
    #Optimization-based metrics.
    #These are testing leximax-Phragmen, core efficiency, and var-Phragmen.
    append!(metric_keys, ["Maximin Support", "Stable Price", "Dispersion Price"])

    #Geometry-based metrics. 
    #These look at the distributional match of voters and candidates.
    #Utility is angular, not linear.
    append!(metric_keys, ["Effective Rank Reduction Ratio", "Expected Cosine Distance",
    "Variance Cosine Distance", "Expected Euclidean Distance", "Variance Euclidean Distance"])

    #Utility-based metrics
    #These characterize the distribution of utilities for winners among voters
    append!(metric_keys, ["Expected Linear Utility", "Variance Linear Utility", 
    "Expected Log Utility", "Variance Log Utility", "Expected Max Utility",
    "Variance Max Utility", "Expected Winner Polarization", "Least Polarized Winner",
    "Most Polarized Winner", "Variance Winner Polarization"])
    
    #Other heuristics
    append!(metric_keys, ["Most Blocking Loser Capture", "Sortition Efficiency", 
    "Average Utility Gain Extra Winner"])


    #=
    plot a distribution of *sample means* over *parameter sweeps*
    aka, individual trials are worthless!

    plot metric vs strategic %. see how it degrades!
    size of dot : subjective realisticness (or maybe # winners?)
    color of dot : voting method
    opacity of dot : variance

    40 samples (each rule + stratified sortition committee)
    aggregate into 1 point per rule, divide by average of random committees

    on top row, have a color bar with the best rule for that strategy level by histogram
    also, have a color bar with rule closest (KS test) to stratified sortition by strategy hist

    on right column, list voting rules with table of:
    mean (over hyperparameter means)
    degradation slope (with linear regression and variance fading)
    borda score (compared to all other rules) (over hyperparam means)
    KS test compared to stratified sorition (aggregate all datapoints)
    fraction of top color bars occupied
    composite quality ranking

    or instead of 2d scatter, can do 2d heatmap! each hyperparam sample has 20 trials, use trimean

    also look as amount of noise increases, alongside as amount of strategy increases
    =#

    # write a function that gives a random-normalized observation

    trials = 100
    g = DataFrame()

    stables = zeros(trials, 4)
    mms = zeros(trials, 4)

    sss_v_mes = 0

    for i in 1:trials
        println()
        println("i = ", i)
        V = 5000
        C = 50
        k = 9

        #S = telos_linf_unichlet(V, C)
        if rand() < 0.6
            println("model is square")
            S = square_euclid_uniform(V, C)
        
        elseif rand() < 0.7
            println("model is cube polar")
            S = cube_polar_uniform(V, C)
        else
            S = uniform_ic(V, C)
        end
        #display(S)
        S = normalize_scores(S)


        w1 = single_transferable_vote(S, k)
        println("STV gives $(w1)")

        w3 = allocated_score(S, k)
        println("AS gives $(w3)")

        w4 = spent_score_accumulation(S, k)
        println("SSS gives $(w4)")

        w2 = method_equal_shares(S, k)
        println("MES gives $(w2)")

        ckp, bw = cdf_transform(S)


        z = 1
        for w in (w1, w2, w3, w4)
            gm = get_metrics(S, w, V, C, k)
        end

        if (committee_is_preferred(S, w2, w4))
            sss_v_mes += 1
        elseif committee_is_preferred(S, w4, w2)
            sss_v_mes -= 1
        end
        #println(committee_is_preferred(S, w4, w2))

        # m = get_metrics(S, w1, V, C, k)
        # if isempty(g)
        #     g = DataFrame(m)
        # else
        #     push!(g, m)
        # end
    end
    # println("PREFERRED MES TO SSS BY A MARGIN OF ", sss_v_mes)
    # println()

    # display(stables)
    # println()
    # display(mms)
    # println()

    # display(mean.(eachcol(stables)))
    # println()
    # display(mean.(eachcol(mms)))
end
main()