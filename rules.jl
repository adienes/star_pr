#export STN, AS, SSS, SMR, EAZ, MES_ExLR, MES_ExMQ, STV, SDV, SSQ
using SpecialFunctions
using Combinatorics

include("metrics/metrics.jl")
function H(utils)
    return digamma(1 + sum(utils)) + (1 - digamma(2))
end
#==============================================================#



#==============================================================#
struct Tally
    budgets::Vector{Float64}
    S::Matrix{Float64}
    T::Matrix{Float64}
    q::Float64
    k::Int
    wset::Vector{Int}

    function Tally(S, q, k)
        new(ones(size(S, 1)), S, copy(S), q, k, [])
    end
end

function harmopt(S, k; q=nothing)
    besthscore = 0
    bestcom = nothing
    for potential_committee in combinations(1:size(S, 2), k)
        if harmonicutil(S, potential_committee) > besthscore
            besthscore = harmonicutil(S, potential_committee)
            bestcom = potential_committee
        end
    end
    return bestcom
end

function MMZ(S, k; q=nothing)
    C = size(S, 2)
    wset::Vector{Int} = []
    for _ = 1:k
        bestmm = 0
        bestcand = 0
        for j = 1:C
            j ∈ wset && continue
            val = maximinsupport(S, [wset; j])
            if val > bestmm
                bestmm = val
                bestcand = j
            end
        end
        push!(wset, bestcand)
    end
    return wset
end

function BS(S, k)
    return collect(partialsortperm(vec(sum(S; dims=1)), 1:k; rev=true))
end

function STN(S, k)
    #note that clones are allowed
    dictators = rand(1:size(S, 1), k)
    return [argmax(S[i, :] for i in dictators)]
end

function SDV(S, k; A=1, B=0, E=1, q=nothing)
    #default params are Jefferson RRV
    #for Webster SDV, set A=0, B=1, E=2
    V,C = size(S)
    T = copy(S)

    satsum::Vector{Float64} = zeros(V)
    wset::Vector{Int} = []
    for _ in 1:k
        w = argmax(linearutilities(1, T))
        push!(wset, w)

        satsum += S[:, w]
        for j in 1:C
            for i in 1:V
                T[i, j] = S[i, j] * (A + B*S[i, j])/(A + B*S[i, j] + E*satsum[i])
            end
        end
        T[:, wset] .= 0
    end

    return wset
end

function SSQ(S, k)
    V,C = size(S)
    T = copy(S)

    satsum::Vector{Float64} = zeros(V)
    wset::Vector{Int} = []
    for _ in 1:k
        w = argmax(linearutilities(1, T))
        push!(wset, w)

        satsum .+= S[:, w]
        for j in 1:C
            denom = 1 + sum(S[:, j] .* satsum) / sum(satsum)
            for i in 1:V
                T[i, j] = S[i, j] / denom
            end
        end
        T[:, wset] .= 0
    end

    return wset
end

function STV(S, k; q=fld(size(S, 1), (k+1))+1)
    #error("bad implementatin")
    V,C = size(S)
    budgets::Vector{Float64} = ones(V)
    wset::Vector{Int} = []
    remaining::Vector{Int} = collect(1:C)

    for _ in 1:k
        fp = firstpreferences(budgets, S[:, remaining])
        while (maximum(fp) < q) && length(remaining) > k - length(wset)
            loser = remaining[argmin(fp)]
            filter!(j -> j != loser, remaining)
            fp = firstpreferences(budgets, S[:, remaining])
        end

        wix = argmax(fp)

        subs = S[:, remaining]
        preferred = (subs.== mapslices(x -> maximum(x), subs, dims=2))
        winnercontribution = budgets .* (preferred ./ sum(preferred, dims=2))[:, wix]
        surplus = max(sum(winnercontribution) / q, 1)
        budgets .-= (winnercontribution ./ surplus)

        w = remaining[wix]
        push!(wset, w)
        filter!(j -> j != w, remaining)
    end
    return wset
end

function AS(S, k; q=fld(size(S, 1), (k+1))+1, current=true)
    AS_select(tally::Tally) = argmax(linearutilities(tally.budgets, tally.T))
    AS_reweight!(tally::Tally, w) = allocate!(tally.budgets, tally.T[:, w], q; current=current)
    return sequentially(S, q, k, AS_select, AS_reweight!)
end

function SSS(S, k; q=fld(size(S, 1), (k+1))+1)
    SSS_select(tally::Tally) = argmax(linearutilities(tally.budgets, tally.T))
    SSS_reweight!(tally::Tally, w) = enphrlinear!(tally.budgets, tally.T[:, w], q)
    return sequentially(S, q, k, SSS_select, SSS_reweight!)
end

function SMR(S, k; q=fld(size(S, 1), (k+1))+1)
    function SMR_select(tally::Tally)
        (; budgets, T, q) = tally
        monroeutilities = map((j) -> monroeutility(budgets, j, q), eachcol(T))
        tied = findall(monroeutilities .== maximum(monroeutilities))
        return tied[argmax(linearutilities(budgets, T[:, tied]))]        

    end
    SMR_reweight!(tally::Tally, w) = allocate!(tally.budgets, tally.T[:, w], q)
    return sequentially(S, q, k, SMR_select, SMR_reweight!)
end

function EAZ1(S, k; q=fld(size(S, 1), (k+1))+1)
    function EAZ_select(tally::Tally)
        (; budgets, T, q) = tally
        rhos = map((j) -> bucklinprice(budgets, j, q), eachcol(T))
        ρ = maximum(rhos)
        tied = findall(rhos .== ρ)
        ρ == 0 && begin ρ += eps() end
        return tied[argmax(linearutilities(budgets, T[:, tied] .≥ ρ))]
    end
    function EAZ_reweight!(tally::Tally, w)
        (; budgets, T, q) = tally
        ρ = bucklinprice(budgets, T[:, w], q)
        ρ == 0 && begin ρ += eps() end
        enphrlinear!(budgets, T[:, w] .≥ ρ, q)
    end
    return sequentially(S, q, k, EAZ_select, EAZ_reweight!)
end

function EAZ2(S, k; q=fld(size(S, 1), (k+1))+1)
    function EAZ_select(tally::Tally)
        (; budgets, T, q) = tally
        rhos = map((j) -> bucklinprice(budgets, j, q), eachcol(T))
        ρ = maximum(rhos)
        tied = findall(rhos .== ρ)
        ρ == 0 && begin ρ += eps() end
        return tied[argmax(linearutilities(budgets, T[:, tied] .≥ ρ))]
    end
    function EAZ_reweight!(tally::Tally, w)
        (; budgets, T, q) = tally
        ρ = bucklinprice(budgets, T[:, w], q)
        ρ == 0 && begin ρ += eps() end
        spendequally!(budgets, T[:, w] .≥ ρ, q)
    end
    return sequentially(S, q, k, EAZ_select, EAZ_reweight!)
end

function DRB1(S, k; q=fld(size(S, 1), (k+1))+1)
    #linear select, bucklin mes reweight
    function DRB_combined!(tally::Tally)
        (; budgets, T, q) = tally
        w = argmax(linearutilities(budgets, T))
        d = bucklinprice(budgets, T[:, w], q)
        iszero(d) && begin d += eps() end

        spendequally!(budgets, (T .≥ d)[:, w], q)
        return w
    end
    return sequentially(S, q, k, DRB_combined!, (::Tally, ::Int)->nothing)
end

function DRB2(S, k; q=fld(size(S, 1), (k+1))+1)
    #runoff only when to exhaust
    function DRB_combined!(tally::Tally)
        (; budgets, T, q) = tally
        w = argmax(linearutilities(budgets, T))
        d = bucklinprice(budgets, T[:, w], q)

        if iszero(d)
            a,b = partialsortperm(linearutilities(budgets, T), 1:2; rev=true)
            m = majoritymargin(budgets, T[:, a], T[:, b])
            if m > 0
                w = a
            else
                w = b
            end
            d = bucklinprice(budgets, T[:, w], q)
            d == 0 && begin d += eps() end
        end

        spendequally!(budgets, (T .≥ d)[:, w], q)
        return w
    end
    return sequentially(S, q, k, DRB_combined!, (::Tally, ::Int)->nothing)
end

function DRB3(S, k; q=fld(size(S, 1), (k+1))+1)
    #runoff every seat
    function DRB_combined!(tally::Tally)
        (; budgets, T, q) = tally
        w = 0
        a,b = partialsortperm(linearutilities(budgets, T), 1:2; rev=true)
        m = majoritymargin(budgets, T[:, a], T[:, b])
        if m > 0
            w = a
        else
            w = b
        end
        d = bucklinprice(budgets, T[:, w], q)
        iszero(d) && begin d += eps() end
        spendequally!(budgets, (T .≥ d)[:, w], q)
        return w
    end
    return sequentially(S, q, k, DRB_combined!, (::Tally, ::Int)->nothing)
end


function MES_ExLR(S, k; q=fld(size(S, 1), (k+1))+1)
    function MES_ExLR_select(tally::Tally)
        (; budgets, T, q) = tally
        rhos = map((j) -> uniformprice(budgets, j, q), eachcol(T))
        if isinf(minimum(rhos))
            w = argmax(linearutilities(budgets, T))
        else
            w = argmin(rhos) #assume ties rare
        end
        return w
    end
    MES_ExLR_reweight!(tally::Tally, w) = spendequally!(tally.budgets, tally.T[:, w], tally.q)
    return sequentially(S, q, k, MES_ExLR_select, MES_ExLR_reweight!)
end

function MES_ExMQ(S, k; q=fld(size(S, 1), (k+1))+1)
    function MES_ExMQ_select(tally::Tally)
        (; budgets, T, q) = tally
        rhos = map((j) -> uniformprice(budgets, j, q), eachcol(T))
        if isinf(minimum(rhos))
            #If the weighted sum is a quota then the candidate will be affordable at rho=1
            surplus = maximum(linearutilities(budgets, T)) / q
            budgets ./= surplus
        end
        rhos = map((j) -> uniformprice(budgets, j, q), eachcol(T))
        w = argmin(rhos)
        @assert(w < Inf)
        return w
    end
    MES_ExMQ_reweight!(tally::Tally, w) = spendequally!(tally.budgets, tally.T[:, w], tally.q)
    return sequentially(S, q, k, MES_ExMQ_select, MES_ExMQ_reweight!)
end

#=
Below are helper functions
_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_
=#

function sequentially(S, q, k, select, reweight!)
    tally = Tally(S, q, k)
    for _ in 1:k
        w = select(tally)
        reweight!(tally, w)
        push!(tally.wset, w)
        tally.T[:, w] .= 0
    end

    return tally.wset
end

function majoritymargin(budgets, ui, uj)
    ipref = ui .> uj
    jpref = uj .> ui
    return sum(budgets .* ipref) - sum(budgets .* jpref)
end

function uniformprice(budgets, utils, q)
    support = findall(utils .> 0)
    sortexhprice = sort(support, by = i -> budgets[i] / utils[i])

    rescost = q
    resutil = sum(utils[support])
    
    ρ = Inf
    sum(budgets[utils .> 0]) < rescost && return ρ

    for i in sortexhprice
        #If this is false, then i is not exhausted and cand is electable
        #at rho = rescost / resutil
        rescost * utils[i] < budgets[i] * resutil && break

        #Raise rho to exhaust i
        rescost -= budgets[i]
        resutil -= utils[i]
    end

    if isapprox(resutil+1, 1) || isapprox(rescost+1, 1)
        #needed in edge exact-quota cases.
        #technically the condition should be perfectly correlated.
        ρ = budgets[last(sortexhprice)] / utils[last(sortexhprice)]
    else
        ρ = rescost / resutil
    end

    return ρ
end

function bucklinprice(budgets, utils, q)
    suppsorted = sortperm(utils, rev=true)
    edgevoter = findfirst(r -> r ≥ q, cumsum(budgets[suppsorted]))
    if isnothing(edgevoter)
        edgeutil = 0
    else
        edgeutil = utils[suppsorted[edgevoter]]
    end
    return edgeutil
end

function monroeutility(budgets, utils, q)
    suppsorted = sortperm(utils, rev=true)
    edgevoter = findfirst(r -> r ≥ q, cumsum(budgets[suppsorted]))
    isnothing(edgevoter) && return sum(budgets .* utils)
    return sum((budgets .* utils)[suppsorted[1:edgevoter]])
end

function linearutilities(budgets, T)
    return vec(sum(budgets .* T, dims=1))
end

function firstpreferences(budgets, T)
    #0-1 array of where voter's current support is
    preferred = (T .== mapslices(x -> maximum(x), T, dims=2))

    #what fraction of a voter's remaining ballot weight should be assigned
    relativecontribution = preferred ./ sum(preferred, dims=2)

    return vec(sum(budgets .* relativecontribution, dims=1))
end

function allocate!(budgets, utils, q; current=true)
    uwrk = current ? budgets.*utils : utils
    edgeutil = bucklinprice(budgets, uwrk, q)

    #if a quota is not reached, largest remainders
    votersabove = (uwrk .> edgeutil)
    if edgeutil > 0
        voterson = (uwrk .== edgeutil)
        surplus = sum(budgets[voterson]) / (q - sum(budgets[votersabove]))
        budgets[voterson] .*= (surplus - 1)/surplus
    end
    budgets[votersabove] .= 0
end

function spendequally!(budgets, utils, q)
    ρ = uniformprice(budgets, utils, q)
    budgets .= max.(0, budgets .- (ρ.*utils))
end

function enphrlinear!(budgets, utils, q)
    uwrk = budgets.*utils
    surplus = max(sum(uwrk) / q, 1)
    budgets .= max.(0, (budgets - uwrk ./ surplus))
end