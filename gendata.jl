using DataFrames
using CSV

include("./rules.jl")
include("./metrics/metrics.jl")
include("./strategy/strategy.jl")
include("./models/mallows.jl")
include("./models/spatial.jl")

function main()
    methodnames = Dict(
		"STN" => STN,
		"AS" => AS,
		"SSS" => SSS,
		"SMR" => SMR,
		"MES_ExLR" => MES_ExLR,
		"MES_ExMQ" => MES_ExMQ,
		"STV" => STV,
		"SDV" => SDV,
		"SSQ" => SSQ
	)
	
	metricnames = Dict(
		"Average Linear Utility" => meanlinutil,
		"Variance Linear Utility" => varlinutil,
		"Average Maximum Utility" => meanmaxutil,
		"Variance Maximum Utility" => varmaxutil,
		"Nash Product Welfare" => meanlogutil,
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
		"numwinners" => Int[]
		),
		Dict(
			[s => Float64[] for s in keys(metricnames)]
		)
	))

	numvoters = 2000
	numtrials = 20
	numscores = 9
	#Janky, but Mallows C is fixed right now.

	for t in 1:numtrials
        rand() < 0.08 && println(t)
		poller = SpatialElection{2}(0, numvoters, 40)
		#poller = MallowsElection(5, numvoters, 40)
		cast!(poller)
		S_incere = round.(poller.S .* numscores) ./ numscores
		S_incere = normalizescores(S_incere)
		
		for m in keys(methodnames), k in [5]
			m in ["STN", "SMR"] && continue
			point = Dict(
				"hpuid" => poller.E.hpuid,
				"stratpct" => poller.E.ω,
				"numwinners" => k,
				"methodname" => m
			)

			ws_incere = methodnames[m](S_incere, k)

			stratvoters = collect(1:numvoters)[poller.strategicbehavior]
			S_trat = frontrunner_bullets(S_incere, ws_incere; voters = stratvoters)

			ws_trat = methodnames[m](S_trat, k)

			metricset = Dict(
				[s => metricnames[s](S_incere, ws_trat) for s in keys(metricnames)]
			)
			merge!(point, metricset)
			
			push!(df, point)
		end

        # for sortitiontrials in 1:4, k in [5]
        #     point = Dict(
		# 		"hpuid" => poller.E.hpuid,
		# 		"stratpct" => poller.E.ω,
		# 		"numwinners" => k,
		# 		"methodname" => "STN"
		# 	)
			

		# 	ws_incere = methodnames["STN"](S_incere, k)
		# 	metricset = Dict(
		# 		[s => metricnames[s](S_incere, ws_incere) for s in keys(metricnames)]
		# 	)
		# 	merge!(point, metricset)
			
		# 	push!(df, point)
        # end
	end
    CSV.write("smalldroop.csv", df)
end
main()