# -*- coding: utf-8 -*-
using LinearAlgebra
using Statistics
# using CUDA
using PyCall
sklearn_linear_model = pyimport("sklearn.linear_model")

"""
function lasso_path(X, y; λs = nothing, P = 100, eps = 1e-3, maxIter = 1000, 
                    tol = 1e-4, method = :Od)

Find β that minimises 1/2*||y-Xβ||₂² + λ||β||₁ for several λs.

It assumes that y has zero mean.
It may also be benefitial to normalise X and y to have zero mean and unit variance.

X : matrix (N,d) of floats, with N inputs of d dimension.
y : array of N floats, with N outputs.
λs : array of P floats, with regularisation parameters to use. If nothing, they will be chosen automatically.
P : int, number of λs to try, if λs are chosen automatically.
eps : float, ratio λmin/λmax if λs are chosen automatically.
maxIter : int, the maximum number of iterations of coordinate descent.
tol : float, tolerance used to check whether to stop iterations.
The stopping criterion is `if maximum(abs.(β .- oldβ)) < tol * maximum(abs.(β)) break`.
method : either :Od, :On, :uncorrelated, :SK.
In case :Od, each iteration of coordinate_descent! has complexity O(d^2). 
In case :On, each iteration has complexity O(nd). Better when n<<d. (in theory...)
In case :uncorrelated, it is assumed that cov(X) is zero everywhere except the diagonal,
and then only one step is required per dimension, without any iterations.
Some multithreaded options: :Asy and :wild
In case :SK, the sklearn.lasso_path() method is used. This method has its own calculation
of λs.

Returns λs, βs, nIters
"""
function lasso_path(X, y; λs = nothing, P = 100, eps = 1e-3, maxIter = 1000, tol = 1e-4, method = :Od)
    N,d = size(X)

    # Choose λs if not provided.
    if λs == nothing
        λmax = maximum(abs.(X' * y))
        λs = [λmax * eps^(p/P) for p=0:P-1]
    end
    @assert all((λs[1:end-1] .- λs[2:end]) .> 0.0) "λs must be in descending order."
    P = length(λs)
    βs = zeros(Float64, P, d)
    nIters = zeros(Int64, P)
    
    # Lasso path:
    if method == :Od
        XX = X' * X
        Xres = X' * y  # X'*residue = X'*(y - X*β)
        for i=2:P
            βs[i,:] .+= βs[i-1,:]  # Warm start for next λ.
            nIters[i] = coordinate_descent!(XX, Xres, λs[i], view(βs, i, :), 
                    maxIter = maxIter, tol = tol)
        end
    elseif method == :On
        XXmin1 = sum(X.^2, dims=1) .^ (-1)
        res = copy(y)  # Residue y - X*β
        for i=2:P
            βs[i,:] .+= βs[i-1,:]  # Warm start for next λ.
            nIters[i] = coordinate_descent!(XXmin1, res, λs[i], view(βs, i, :), X,
                    maxIter = maxIter, tol = tol)
        end
    elseif method == :uncorrelated
        XXmin1 = sum(X.^2, dims=1) .^ (-1)
        Xy = X' * y
        Threads.@threads for i=1:P
            uncorrelated_coordinate_descent!(XXmin1, Xy, λs[i], view(βs, i, :))
        end
    elseif method == :wild
        XXmin1 = sum(X.^2, dims=1) .^ (-1)
        res = copy(y)  # Residue y - X*β
        for i=2:P
            βs[i,:] .+= βs[i-1,:]  # Warm start for next λ.
            nIters[i] = wild_coordinate_descent!(XXmin1, res, λs[i], view(βs, i, :), X,
                    maxIter = maxIter, tol = tol)
        end
    elseif method == :Asy
        XX = X'*X
        for i=2:P
            βs[i,:] .+= βs[i-1,:]  # Warm start for next λ.
            nIters[i] = Asy_coordinate_descent!(XX, X, y, λs[i], view(βs, i, :), 
                    maxIter = maxIter, tol = tol)
        end
    elseif method == :SK
        skresults = sklearn_linear_model.lasso_path(X,y)
        βs = skresults[2]'
        λs = skresults[1]
    else
        println("Error, unknown method $method !!")
        return 1
    end
    
    return λs, βs, nIters
end


"""
function normalise!(Xs...)

Normalises each matrix in Xs by subtracting mean and
dividing by variance of Xs[1].
"""
function normalise!(Xs...)
    μ = mean(Xs[1], dims = 1)
    σ = std(Xs[1], dims = 1)
    for X in Xs
        X .-= μ; X ./= σ
    end
end


"""
function mse(y,ypred)

Mean squared error.
y and ypred are arrays of floats.
"""
mse(y,ypred) = mean((y-ypred).^2)


"""
function fast_square(X)

Computes X'*X. Uses symmetry arguments
to avoid repeating inner products.

Returns X'*X
"""
function fast_square(X)
    N,d = size(X)
    XX = zeros(d,d)
    @simd for i=1:d
        @inbounds XX[i,i:d] = sum(X[1:N,i] .* X[1:N,i:d], dims=1)
    end
    return XX
end


"""
function coordinate_descent!(XX, Xres, λ, β; maxIter = 1000, tol = 1e-4)

Coordinate descent algorithm to find β that minimises 1/2*||y-Xβ||₂² + λ||β||₁,
where X has shape (N,d) and contains N d-dimensional inputs, and 
y is an array of length N containing N outputs.
It assumes that y has zero mean.
It may also be benefitial to normalise X and y to have zero mean and unit variance.
This version of coordinate descent is O(d^2) at each iteration.

XX : matrix (d,d) of floats, obtained with X'*X 
Xres : array of floats of length d, obtained with X'*(y-X*β). Xres stands for X*residue.
λ : float, regularisation parameter.
β : array of floats of length d, initial guess for β. If unknown, choose zeros.
At the end β contains the found optimal coefficients.
maxIter : int, the maximum number of iterations of coordinate descent.
tol : float, tolerance used to check whether to stop iterations.
The stopping criterion is `if maximum(abs.(β .- oldβ)) < tol * maximum(abs.(β)) break`.

TODO
feat_sel : :cyclic or :random. When random, at each iteration the algorithm uses a new seed
to randomly iterate through the elements of β, rather than updating them in order.
seed : int, seed used if feat_sel=:cyclic.

Returns the number of iterations required.
"""
function coordinate_descent!(XX, Xres, λ, β; maxIter = 1000, tol = 1e-4)
    oldβ = 0.0
    maxChange = 0.0
    maxβ = 0.0
    iter = 0
    
    while (iter < maxIter) && (maxChange >= tol*maxβ)
        iter += 1
        maxChange = 0.0
        maxβ = 0.0
        for j in eachindex(β)
            oldβ = β[j]
            if β[j] != 0.0 Xres .+= β[j] .* XX[:,j] end
            β[j] = max(abs(Xres[j]) - λ, 0.0)
            if β[j] != 0.0 
                β[j] *= sign(Xres[j]) / XX[j,j]
                Xres .-= β[j] .* XX[:,j]
                maxβ = max(abs(β[j]), maxβ)
            end
            maxChange = max(abs(β[j] - oldβ), maxChange)
        end
    end
    return iter
end

"""
function coordinate_descent!(XXmin1, res, λ, β, X::Array{Float64}; maxIter = 1000, tol = 1e-4)

Coordinate descent but with O(dn) complexity per iteration. This implementation is the same 
used in the SKlearn coordinate descent algorithm.

XXmin1 : array of floats of length d, obtained with 1 ./ diag(X'*X)
res : array of floats of length N, obtained with (y-X*β). res stands for residue.
X : matrix (N,d) of floats, with N inputs of d dimension.

The rest is the same as the normal coordinate descent
"""
function coordinate_descent!(XXmin1, res, λ, β, X::Array{Float64}; maxIter = 1000, tol = 1e-4)
    oldβ = 0.0
    maxChange = 0.0
    maxβ = 0.0
    iter = 0
    tmp = 0.0
    
    while (iter < maxIter) && (maxChange >= tol*maxβ)
        iter += 1
        maxChange = 0.0
        maxβ = 0.0
        for j in eachindex(β)
            oldβ = β[j]
            if β[j] != 0.0 res .+= β[j] .* X[:,j] end
            tmp = X[:,j] · res
            β[j] = max(abs(tmp) - λ, 0.0)
            if β[j] != 0.0 
                β[j] *= sign(tmp) * XXmin1[j]
                res .-= β[j] .* X[:,j]
                maxβ = max(abs(β[j]), maxβ)
            end
            maxChange = max(abs(β[j] - oldβ), maxChange)
        end
    end 
    return iter
end

"""
function wild_coordinate_descent!(XXmin1, res, λ, β, X::Array{Float64}; maxIter = 1000, tol = 1e-4)

Multithreaded coordinate descent, with the "wild" updated of the shared vector.
"""
function wild_coordinate_descent!(XXmin1, res, λ, β, X::Array{Float64}; maxIter = 1000, tol = 1e-4)
    oldβ = zeros(length(β))  # TODO: These two vectors could have length nthreads() to save memory.
    tmp = zeros(length(β))
    maxChange = 0.0
    maxβ = 0.0
    iter = 0

    while (iter < maxIter) && (maxChange >= tol*maxβ)
        iter += 1
        # Parallel coordinate update:
        Threads.@threads for j in eachindex(β)
            oldβ[j] = β[j]
            tmp[j] = X[:,j] · res
            if β[j] != 0.0 tmp[j] += β[j] * dot(X[:,j], X[:,j]) end
            β[j] = max(abs(tmp[j]) - λ, 0.0)
            if β[j] != 0.0 β[j] *= sign(tmp[j]) * XXmin1[j] end
            # Wild update of shared vector:
            for i in eachindex(res)  # Warning! loop has race conditions! (hence the "wild")
                res[i] += X[i,j] * (oldβ[j] - β[j])
            end
        end
    
        maxβ = maximum(abs.(β))
        maxChange = maximum(abs.(β .- oldβ))
    end 
    return iter
end


"""
function Asy_coordinate_descent!(XX, X, y, λ, β; maxIter = 1000, tol = 1e-4)

AsySCD. Multithreaded coordinate descent without shared vector, only updates to β.
It's possible that each thread uses an outdated version of β due to multithreading.
"""
function Asy_coordinate_descent!(XX, X, y, λ, β; maxIter = 1000, tol = 1e-4)
    oldβ = zeros(length(β))  # TODO: These two vectors could have length nthreads() to save memory.
    tmp = zeros(length(β))
    maxChange = 0.0
    maxβ = 0.0
    iter = 0

    while (iter < maxIter) && (maxChange >= tol*maxβ)
        iter += 1
        # Parallel coordinate update:
        Threads.@threads for j in eachindex(β)
            oldβ[j] = β[j]
            β[j] = 0.0
            tmp[j] = X[:,j]·y - XX[:,j]·β
            β[j] = max(abs(tmp[j]) - λ, 0.0)
            if β[j] != 0.0 β[j] *= sign(tmp[j]) / XX[j,j] end
        end
        maxβ = maximum(abs.(β))
        maxChange = maximum(abs.(β .- oldβ))
    end 
    return iter
end


"""
function uncorrelated_coordinate_descent!(XXmin1, Xy, λ, β)

Under the assumption that X is orthogonal (cov(X) zero everywhere but the diagonal)
performs fast coordinate descent without iterations.

XXmin1 : array of floats of length d, obtained with 1 ./ diag(X'*X)

"""
function uncorrelated_coordinate_descent!(XXmin1, Xy, λ, β)

    β = sign.(Xy) .* max.(abs.(Xy) .- λ, 0.0)
    for j in eachindex(β)
        if β[j] != 0.0
            β[j] *= XXmin1[j]
        end
    end
end

