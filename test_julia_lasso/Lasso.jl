using LinearAlgebra
using Statistics
# using CUDA
# using PyCall
# sklearn_linear_model = pyimport("sklearn.linear_model")

"""
function lasso_path(X, y; λs = nothing, P = 100, eps = 1e-3, maxIter = 1000, tol = 1e-4)

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
In case :SK, the sklearn.lasso_path() method is used. This method has its own calculation
of λs.

Returns λs, βs, nIters
"""
function lasso_path(X, y; λs = nothing, P = 100, eps = 1e-3, maxIter = 1000, tol = 1e-4, method = :Od)
    # Prepare variables.
    N,d = size(X)
    V = nothing
    if method == :uncorrelated_CUDA
        X = CuArray(X)
        y = CuArray(y)
    end
    if (method == :uncorrelated_CUDA) || (method == :uncorrelated)
        # Perform PCA
        _,S,V = svd(cov(X))
        V = V[:,1:min(N,d)]
        projection = V
        X = X * V * diagm(S .^ (-0.5))
        X = X * V
        d = size(X)[2]
    end
    XX = X' * X; res = copy(y); Xres = X' * y  # X*residue, with residue=y-Xβ
    
    # Choose λs if not given.
    if λs == nothing
        λmax = maximum(abs.(Xres))
        λs = [λmax * eps^(p/P) for p=0:P-1]
    end
    @assert all((λs[1:end-1] .- λs[2:end]) .> 0.0) "λs must be in descending order."
    P = length(λs)
    βs = zeros(Float64, P, d)
    nIters = zeros(Int64, P)
    
    # Lasso path:
    if method == :Od
        for i=2:P
            βs[i,:] .+= βs[i-1,:]  # Warm start for next λ.
            nIters[i] = coordinate_descent!(XX, Xres, λs[i], view(βs, i, :), 
                    maxIter = maxIter, tol = tol)
        end
    elseif method == :On
        XXmin1 = diag(XX) .^ (-1)
        for i=2:P
            βs[i,:] .+= βs[i-1,:]  # Warm start for next λ.
            nIters[i] = coordinate_descent!(XXmin1, res, λs[i], view(βs, i, :), X,
                    maxIter = maxIter, tol = tol)
        end
    elseif method == :uncorrelated
        XXmin1 = diag(XX) .^ (-1)
        Threads.@threads for i=1:P
            uncorrelated_coordinate_descent!(XXmin1, Xres, λs[i], view(βs, i, :))
        end
    elseif method == :uncorrelated_CUDA
        βs = CuArray(βs)
        λs = CuArray(λs)
        XXmin1 = diag(XX) .^ (-1)
        for i=1:P
            CUDA_uncorrelated_coordinate_descent!(XXmin1, Xres, λs[i], view(βs, i, :))
        end
    elseif method == :SK
        skresults = sklearn_linear_model.lasso_path(X,y)
        βs = skresults[2]'
        λs = skresults[1]
    else
        println("Error, unknown method $method !!")
        return 1
    end

    if (method == :uncorrelated_CUDA) || (method == :uncorrelated)
        X = X * V'
        βs = βs * V'
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

mse(y,ypred) = mean((y-ypred).^2)


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
    d = length(β)
    
    while (iter < maxIter) && (maxChange >= tol*maxβ)
        iter += 1
        maxChange = 0.0
        maxβ = 0.0
        for j=1:d
            oldβ = β[j]
            if β[j] != 0.0 Xres .+= β[j] .* XX[1:d,j] end
            β[j] = max(abs(Xres[j]) - λ, 0.0)
            if β[j] != 0.0 
                β[j] *= sign(Xres[j]) / XX[j,j]
                Xres .-= β[j] .* XX[1:d,j]
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
    d = length(β)
    N = length(res)
    
    while (iter < maxIter) && (maxChange >= tol*maxβ)
        iter += 1
        maxChange = 0.0
        maxβ = 0.0
        for j=1:d
            oldβ = β[j]
            if β[j] != 0.0 res .+= β[j] .* X[1:N,j] end
            tmp = X[1:N,j] · res
            β[j] = max(abs(tmp) - λ, 0.0)
            if β[j] != 0.0 
                β[j] *= sign(tmp) * XXmin1[j]
                res .-= β[j] .* X[1:N,j]
                maxβ = max(abs(β[j]), maxβ)
            end
            maxChange = max(abs(β[j] - oldβ), maxChange)
        end
    end 
    return iter
end


"""
uncorrelated coordinate descent
"""
function uncorrelated_coordinate_descent!(XXmin1, Xy, λ, β)

    # β = max.(abs.(Xy) .- λ, 0.0)
    for j=1:length(β)
        β[j] = max(abs(Xy[j]) - λ, 0.0)
        if β[j] != 0.0  # This if takes advantage of sparsity.
            β[j] *= sign(Xy[j]) * XXmin1[j]
        end
    end
end

"""
Cuda version of uncorrelated coordinate descent
This version simply doesn't have ifs
The passed arrays can all be CuArrays
"""
function CUDA_uncorrelated_coordinate_descent!(XXmin1, Xy, λ, β)
    β = sign.(Xy) .* (max.(abs.(Xy) .- λ, 0.0)) .* XXmin1
end

