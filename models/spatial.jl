include("./baseparty.jl")

struct SpatialParty{N} <: Party
    μ::Vector{Float64} #center of multivariate gaussian
    σ::Vector{Float64} #stdev along each axis

    λ::Float64 #natural disposition
    ω::Float64 #proportion of strategic voters

    function SpatialParty{N}() where {N}
        μ = (rand(N) + rand(N)) .- 1
        σ = (rand(N)) .* 0.06

        λ = rand()*0.8+0.04
        new{N}(μ, σ, λ)
    end
end

mutable struct SpatialElection{N} <: Election
    E::PartyElectorate

    latentvoters::Vector{Vector{Float64}}
    latentcands::Vector{Vector{Float64}}

    strategicbehavior::Vector{Bool}
    partyaffiliation::Vector{Int}
    
    S::Matrix{Float64}
    function SpatialElection{N}(k, V, C) where {N}
        E = PartyElectorate(k, V, C, randparties(()->SpatialParty{N}()))

        new(E, [zeros(N) for i in 1:V], [zeros(N) for j in 1:C], zeros(V), zeros(V), zeros(V, C))
    end
end

function getvoter(P::SpatialParty)
    return rand.([Normal(x...) for x in zip(P.μ, P.σ)])
end

function getcands(B::SpatialElection; voters=Vector{Float64}[])
    E = B.E
    degressive_population = sqrt.(E.Q)/sum(sqrt.(E.Q))
    party_sampler = Categorical(degressive_population)
    cands = Vector{Float64}[]
    for _ in 1:fld(E.C, 2)
        P = E.L[rand(party_sampler)]
        push!(cands, rand.([Normal(μ, 3.6*σ) for (μ, σ) in zip(P.μ, P.σ)]))
        push!(cands, rand.([Normal(μ, 0.8*σ) for (μ, σ) in zip(P.μ, P.σ)]))
    end
    return cands
end

function ξ(P::SpatialParty, v, c)
    x = euclidean(v,c)
    return max(0, 1 - x/P.λ)
end