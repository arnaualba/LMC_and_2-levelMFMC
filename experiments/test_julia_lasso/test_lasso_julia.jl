0# -*- coding: utf-8 -*-
include("Lasso.jl")
using Printf
using Random

nthread = Threads.nthreads()
println("\nI am using $nthread threads")

seed = abs(rand(Int)); println("Using seed ", seed)
N = 800
d = 10000
println("N = ", N, ", d = ", d)    

function test_Lasso_methods(N, d ;verbose = true)
    X = randn(N,d); Xtest = randn(N,d); normalise!(X,Xtest)
    α = rand(d); α[1:10] = 40.0 .* (rand(10) .- 0.5)  # y = X*α + noise
    noise = 5.5
    y = X * α .+ randn(N)*noise; ytest = Xtest * α; normalise!(y,ytest)
    methods = [
                # :Od,
                # :Od_threaded,
                :no_shared_vector,
                :no_shared_vector_threaded,
                :On,
                :On_threaded, 
                # :uncorrelated,  # Doing an SVD is expensive.
                :SK,
                ]
    ts = []; βs = []; λs = []; mses = []; losses = []
    for method in methods
        t = @elapsed locλs, locβs, nIters = lasso_path(X, y, method = method)
        
        loclosses = [mse(ytest, Xtest*locβs[:,i]) for i=1:length(locλs)]
        push!(ts, t)
        push!(βs, locβs[:,argmin(loclosses)])
        push!(λs, locλs[argmin(loclosses)]) 
        push!(mses, mse(ytest, Xtest*βs[end]))
        push!(losses, loclosses)
    end


    # Print stuff:
    if verbose
        @printf "MSE with %20s = %.5f\n" "zeros" mse(ytest, zeros(length(ytest)))
        OLStime = @elapsed OLSbeta = pinv(X) * y
        @printf "MSE with %20s = %.5f\n" "OLS" mse(ytest, Xtest*OLSbeta)
        for (i,method) in enumerate(methods)
            @printf "MSE with %20s = %.5f\n" string(method) mses[i]
        end
        println()
        @printf "Time with %20s = %.5f s\n" "OLS" OLStime
        for (i,method) in enumerate(methods)
            @printf "Time with %20s = %.5f s\n" string(method) ts[i]
        end
    end
end

# Run once to force compilation:
Random.seed!(seed);
test_Lasso_methods(100, 10, verbose = false)

# Run for real:
Nrep = 1
for n=1:Nrep
    println("\nRepetition $n of $Nrep")
    test_Lasso_methods(N, d, verbose = true)
end


# l = @layout [a b ; c]
# p1 = scatter(string.(methods), ts, 
#     ylabel = "runtime [s]", label = :none, markersize = 10, yscale = :log10)

# p2 = scatter(string.(methods), mses, 
#     ylabel = "MSE", label = :none, markersize = 10, yscale = :log10)
# p2 = hline!([mse(ytest, zeros(length(ytest)))], linestyle=:dash, lw = 4, label = "Baseline MSE")

# p3 = plot()
# colors = [:yellow, :red, :blue, :green]
# shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross]
# λs_for_plot = lasso_path(X, y, maxIter = 1)[1]
# for i=1:length(λs)
#     p3 = scatter!(λs_for_plot, losses[i], lw = 3, label = string(methods[i]),
#         color = colors[i], markershape = shapes[i], markersize = 5)
# end
# p3 = hline!([mse(ytest, zeros(length(ytest)))], linestyle=:dash, lw = 4, 
#     label = "Baseline MSE", xlabel = "λ", ylabel = "MSE", xscale = :log10)

# plot(p1, p2, p3, layout = l, size = (800,600))
