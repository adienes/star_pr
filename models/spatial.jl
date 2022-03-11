include("./baseparty.jl")

struct SpatialParty{N} <: Party
    μ::Vector{Float64} #center of multivariate gaussian
    σ::Vector{Float64} #stdev along each axis

    λ::Float64 #natural disposition
    ω::Float64 #proportion of strategic voters

    function SpatialParty{N}() where {N}
        μ = (rand(N) + rand(N)) .- 1 #slight centrist bias on [-1, +1]
        σ = (rand(N) + rand(N))*0.1 #tip-to-tip about 20% utility

        λ = (rand()*0.6)+0.2
        ω = rand()
        new{N}(μ, σ, λ, ω)
    end
end

mutable struct SpatialElection{N} <: Election
    E::PartyElectorate

    latentvoters::Vector{Vector{Float64}}
    latentcands::Vector{Vector{Float64}}

    strategicbehavior::Vector{Int}
    partyaffiliation::Vector{Int}
    
    S::Matrix{Float64}
    function SpatialElection{N}(k, V, C) where {N}
        E = PartyElectorate(k, V, C, randparties(()->SpatialParty{N}()))

        new(E, [zeros(N) for i in 1:V], [zeros(N) for j in 1:C], zeros(V), zeros(V), zeros(V, C))
    end
end

function getvoter(P::SpatialParty)
    return (rand.([Normal(x...) for x in zip(P.μ, P.σ)]), rand(Bernoulli(P.ω)))
end

function getcands(B::SpatialElection)
    E = B.E
    party_sampler = Categorical(E.Q)
    return [getvoter(E.L[rand(party_sampler)])[1] for j in 1:E.C]
end

function ξ(P::SpatialParty, v, c)
    return 2/(1 + exp(euclidean(v,c)/P.λ))
end