include("rules.jl")
include("metrics/metrics.jl")
using Combinatorics


function partyapprovals(voters, approved, cands)
    a = zeros(cands); a[approved] .= 1
    return repeat(a'; outer=(voters, 1))
end


function ispareto(S, W)
end


#####################################################
function path1()
    cands = 21
    k = 18

    A = partyapprovals(120, [1,2,5], cands)
    B = partyapprovals(120, [1,2,6], cands)
    C = partyapprovals(122, [5,7], cands)
    D = partyapprovals(70, [3,4,6], cands)
    E = partyapprovals(50, [3,4], cands)
    F = partyapprovals(120, [3,4,8], cands)
    G = partyapprovals(121, [8,9], cands)
    H = partyapprovals(52, [7], cands)
    I = partyapprovals(65, [9], cands)

    S = vcat(A, B, C, D, E, F, G, H, I)
    for i in 1:12
        r = zeros(21)
        r[9+i] = 1
        S = vcat(S, repeat(r'; outer=(110, 1)))
    end

    return (S, k)
end

########################################################


function path2()
    cands = 15
    A = partyapprovals(21, [1,2,3,4,5], cands)
    B = partyapprovals(41, [6,7,8,9,10], cands).*0.8 + partyapprovals(41, [11,12,13,14,15], cands)
    C1 = partyapprovals(38, [6,7,8,9,10], cands).*0.6
    C2 = partyapprovals(38, [6,7,8,9,10], cands)

    S1 = vcat(A, B, C1)
    S2 = vcat(A, B, C2)

    return (S1, S2, 5)
end