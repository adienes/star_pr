include("./baseparty.jl")

struct MallowsParty <: Party
    α::Vector{Float64} #reference ranking
    ϕ::Float64 #dispersion probability

    λ::Float64 #natural disposition
    ω::Float64 #proportion of strategic voters

    function MallowsParty(; X = 30, Y = 10)
        #these parameters imply each voter ranks around Y candidates
        #parties on average overlap by Y^2 / X ~3 candidates
        α = shuffle(unique(rand(X, Y)))
        ϕ = (rand() + rand() + rand())/3
        
        λ = sqrt(rand())
        ω = rand()
        new(α, ϕ, λ, ω)
    end
end

mutable struct MallowsElection <: Election
    E::PartyElectorate

    latentvoters::Vector{Vector{Float64}}
    latentcands::Vector{Vector{Float64}}

    strategicbehavior::Vector{Int}
    partyaffiliation::Vector{Int}
    
    S::Matrix{Float64}
    function MallowsElection(k, V, C)
        E = PartyElectorate(k, V, C, randparties(()->MallowsParty()))

        new(E, [zeros(N) for i in 1:V], [zeros(N) for j in 1:C], zeros(V), zeros(V), zeros(V, C))
    end
end

function getvoter(P::MallowsParty)
    r::Vector{Float64} = []
    for (i,c) in enumerate(P.α)
        denom = sum([P.ϕ^i for i in 0:(i-1)])
        pos = rand(Categorical([P.ϕ^(i - j)/denom for j in 1:i]))
        insert!(r, pos, c)
    end
    append!(r, shuffle(setdiff(1:30, r)))

    return (r, rand(Bernoulli(P.ω)))
end

function getcands(B::MallowsElection)
    return union([B.E.L[j].α for j in 1:B.E.M]...)
end

function ξ(P::MallowsParty, v, c)
    rank = findfirst(j -> j == c, v)
    isnothing(rank) && return 0
    return P.λ^(rank-1)
end