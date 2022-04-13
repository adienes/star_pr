include("./baseparty.jl")

struct MallowsParty <: Party
    α::Vector{Int} #reference ranking
    ϕ::Float64 #dispersion probability

    λ::Float64 #natural disposition
    function MallowsParty(; X = 20, Y = 8)
        #these parameters imply each voter ranks around Y candidates
        #parties on average overlap by Y^2 / X candidates
        α = shuffle(unique(rand(1:X, Y)))
        ϕ = (rand() + rand())/2
        
        λ = 1/(2 - rand())
        new(α, ϕ, λ)
    end
end

mutable struct MallowsElection <: Election
    E::PartyElectorate

    latentvoters::Vector{Vector{Int}}
    latentcands::Vector{Int}

    strategicbehavior::Vector{Bool}
    partyaffiliation::Vector{Int}
    
    S::Matrix{Float64}
    function MallowsElection(k, V, C)
        E = PartyElectorate(k, V, C, randparties(()->MallowsParty()))

        new(E, [], [], [], [], Array{Float64}(undef, 0, 0))
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

    return r
end

function getcands(B::MallowsElection; voters=Vector{Float64}[])
    return collect(1:B.E.C)
end

function ξ(P::MallowsParty, v, c)
    rank = findfirst(j -> j == c, v)
    isnothing(rank) && return 0
    return P.λ^(rank-1)
end