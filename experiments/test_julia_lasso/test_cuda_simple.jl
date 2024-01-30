using CUDA
include("lasso.jl")

N = 30
d = 50
lamb = 0.1
X = randn(N,d); Xtest = randn(N,d); normalise!(X,Xtest)
α = rand(d); α[1] = 10.0
noise = 5.5
y = X * α .+ randn(N)*noise; ytest = Xtest * α; normalise!(y,ytest)

# Run once to compile:
XX = ones(5,5); Xres = ones(5); lamb = 1.0; β = ones(5)
coordinate_descent!(XX, Xres, lamb, β)
XX = CuArray(XX); Xres = CuArray(Xres); β = CuArray(β)
coordinate_descent!(XX, Xres, lamb, β)
    
    
# CPU version:
println("CPU")
@time begin
    XX = X'*X; Xres = X'*y; β = fill(1.0f0, d);
    coordinate_descent!(XX, Xres, lamb, β)
end

# CUDA version:
println("GPU")
@time begin
    Xcu = CuArray(X); ycu = CuArray(y)
    XX = Xcu'*Xcu; Xres = Xcu'*ycu; β = CUDA.fill(1.0f0, d);
    coordinate_descent!(XX, Xres, lamb, β)
end    
