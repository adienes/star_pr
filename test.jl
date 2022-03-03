using JuMP
using Ipopt

function main()
    model = Model(Ipopt.Optimizer)

    s = 2 #to win s+1 seats
    @variable(model, 0 <= x <= 1)

    @NLobjective(model, Max, x*(1-x)^s)

    optimize!(model)
    println("at x = $(value(x)), get obj = $(objective_value(model))")

end
main()