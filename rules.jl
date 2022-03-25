#export STN, AS, SSS, SMR, MES_ExLR, MES_ExMQ, STV, SDV, SSQ

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

function STN(S, k)
    #note that clones are allowed
    dictators = rand(1:size(S, 1), k)
    return [argmax(S[i, :] for i in dictators)]
end

function SDV(S, k; A=1, B=0, E=1)
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
        if maximum(fp) ≥ q
            subs = S[:, remaining]
            preferred = (subs.== mapslices(x -> maximum(x), subs, dims=2))
            winnercontribution = (preferred ./ sum(preferred, dims=2))[:, wix]
            surplus = max(sum(winnercontribution) / q, 1)
            budgets .*= (1 - 1/surplus) 
        end

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
        tmr = findall(j -> j == maximum(monroeutilities), monroeutilities)
        if length(tmr) > 1
            w = tmr[argmax(linearutilities(budgets, T[:, tmr]))]
        else
            w = first(tmr)
        end
        return w
    end
    SMR_reweight!(tally::Tally, w) = allocate!(tally.budgets, tally.T[:, w], q; current=false)
    return sequentially(S, q, k, SMR_select, SMR_reweight!)
end

function MES_ExLR(S, k; q=fld(size(S, 1), (k+1))+1)
    function MES_ExLR_select(tally::Tally)
        (; budgets, T, q) = tally
        rhos = map((j) -> uniformprice(budgets, j, q), eachcol(T))
        if isinf(minimum(rhos))
            w = argmax(linearutilities(budgets, T))
        else
            w = argmin(rhos)
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
        push!(tally.wset, w)
        reweight!(tally, w)
        tally.T[:, w] .= 0
    end

    return tally.wset
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

function monroeutility(budgets, utils, q)
    suppsorted = sortperm(utils, rev=true)
    edgevoter = findfirst(r -> r ≥ q, cumsum(budgets[suppsorted]))
    isnothing(edgevoter) && return sum(utils)
    return sum(utils[suppsorted[1:edgevoter]])
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
    suppsorted = sortperm(uwrk, rev=true)
    edgevoter = findfirst(r -> r ≥ q, cumsum(budgets[suppsorted]))

    if isnothing(edgevoter)
        edgeutil = 0
    else
        edgeutil = uwrk[suppsorted[edgevoter]]
    end

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