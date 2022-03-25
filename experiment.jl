include("./metrics/metrics.jl")
include("./models/spatial.jl")
include("./models/mallows.jl")
include("./rules/quota_spending.jl")
include("./rules/other_rules.jl")

using DataFrames
using Distributions
using LinearAlgebra
using Random

methodnames = Dict(
		"MES" => method_equal_shares,
		"SSS" => spent_score_scaling,
		"AS" => allocated_score,
		"EPM" => enestrom_phragmen_margins,
		"STN" => stratified_sortition,
	)
	
	metricnames = Dict(
		"Linear Utility" => meanlinutil,
		"Maximum Utility" => meanmaxutil,
		"Loser Capture" => losercapture
	)

	df = DataFrame(merge(
		Dict(
		"hpuid" => Int[],
		"methodname" => String[],
		"stratpct" => Float64[]
		),
		Dict(
			[s => Float64[] for s in keys(metricnames)]
		)
	))
	
(numwinners, numvoters, numcands) = (5, 2000, 50)
poller = MallowsElection(numwinners, numvoters, numcands)
cast!(poller)

display(poller)
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
