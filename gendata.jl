using DataFrames
using CSV

include("./rules.jl")
include("./metrics/metrics.jl")
include("./strategy/strategy.jl")
include("./models/mallows.jl")
include("./models/spatial.jl")

function main()
    methodnames = Dict(
		#"BS" => BS,
		#"STN" => STN,
		#"AS" => AS,
		"SSS" => SSS,
		#"SMR" => SMR,
		#"MMZ" => MMZ,
		#"EAZ1" => EAZ1,
		"EAZ2" => EAZ2,
		"DRB1" => DRB1,
		#"DRB2" => DRB2,
		#"DRB3" => DRB3,
		#"DRB4" => DRB4,
		#"MES_ExLR" => MES_ExLR,
		"MES_ExMQ" => MES_ExMQ,
		#"PSI" => harmopt,
		"STV" => STV,
		#"SDV" => SDV,
		#"SSQ" => SSQ
	)
	
	metricnames = Dict(
		"Linear Utility" => meanlinutil,
		"Variance Linear Utility" => varlinutil,
		"Maximum Utility" => meanmaxutil,
		"Variance Maximum Utility" => varmaxutil,
		"LogProduct Utility" => logprodutil,
		"Harmonic Utility" => harmonicutil,
		"Most Blocking Loser" => defectingquota,
        "Least Polarized Winner" => minpolarized,
        "Most Polarized Winner" => maxpolarized,
		"Average Polarized Winner" => meanpolarized,
		"Variance Polarized Winner" => varpolarized
	)

	df = DataFrame(merge(
		Dict(
		"hpuid" => Int[],
		"methodname" => String[],
		"stratpct" => Float64[],
		"numwinners" => Int[],
		"Maximin Support" => Float64[],
		"Stable Price" => Float64[],
		"PreferredToRule" => Float64[],
		),
		Dict(
			[s => Float64[] for s in keys(metricnames)]
		)
	))

	numvoters = 2000
	numcands = 200
	numwinners = [31]
	numtrials = 600
	numscores = 9
	#Janky, but Mallows C is fixed right now.

	for t in 1:numtrials
        rand() < 0.08 && println(t)
		poller = SpatialElection{4}(0, numvoters, numcands)
		cast!(poller)
		S = poller.S


		X = round.(S, digits=2)
		#X = 1 ./ mapslices(sortperm, X; dims=2)

		println(map(j -> count(j .> 0), eachcol(X)))
		println(mean(map(i -> count(i .> 0.5), eachrow(X))))
		println("Zeroed cands = $(count(map(j-> count( j .> 0), eachcol(X)) .== 0)), zeroed voters = $(count(map(j-> count( j .> 0), eachrow(X)) .== 0))")
		
		S_incere = round.(S .* numscores) ./ numscores
		S_incere = normalizescores(S_incere)

		
		for m in sort([s for s in keys(methodnames)]), k in numwinners
			hare = numvoters/numcands
			#hare = fld(numvoters, k + 1) + 1

			m in ["STN"] && continue
			point = Dict(
				"hpuid" => poller.E.hpuid,
				"stratpct" => poller.E.ω,
				"numwinners" => k,
				"methodname" => m
			)

			ws_incere = methodnames[m](S_incere, k; q=hare)

			stratvoters = collect(1:numvoters)[poller.strategicbehavior]
			S_trat = S_incere#frontrunner_bullets(S_incere, ws_incere; voters = stratvoters)
			ws_trat = ws_incere# methodnames[m](S_trat, k; q=hare)


			metricset = Dict(
				[s => metricnames[s](S_incere, ws_trat) for s in keys(metricnames)]
			)
			
			metricset["PreferredToRule"] = linearcomparison(S_incere, ws_trat, methodnames["DRB1"](S_trat, k; q=hare))
			metricset["Maximin Support"] = maximinsupport(S_incere, ws_trat)
			metricset["Stable Price"] = 0#mean([coreprice(stochastic_bullets(S_incere), ws_trat) for _ in 1:8])
			merge!(point, metricset)
			
			push!(df, point)
		end
		println(combine(groupby(df, ["methodname", "numwinners"]), ["PreferredToRule", "Linear Utility", "Variance Linear Utility", "Harmonic Utility", "Maximum Utility", "Maximin Support"] .=> (x)->(mean(x), sqrt(var(x))), renamecols=false))
		println()
		println()

	end
    CSV.write("stability.csv", df)
end
main()