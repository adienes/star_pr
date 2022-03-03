#=
Input: a score matrix S, where S[i,j] is the score given to candidate j by voter i
=#


function sdv(S::Array{Float64}, k; A = 0, B = 1, E = 2)
    winners::Vector{Int} = []
    sum_of_satisfied_scores::Vector{Float64} = zeros(size(S)[1])

    V,C = size(S)
    S_wrk = copy(S)

    for _ in 1:k
        w = argmax(sum(S_wrk, dims = 1))[2]
        push!(winners, w)

        sum_of_satisfied_scores += S[:, w]
        for c in 1:C
            for v in 1:V
                S_wrk[v, c] = (S[v, c] * (A + B*S[v, c]))/(A + B*S[v, c] + E*sum_of_satisfied_scores[v])
            end
        end
    end

    return winners
end