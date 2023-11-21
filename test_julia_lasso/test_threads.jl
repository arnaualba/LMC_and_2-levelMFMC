using BenchmarkTools
    
nthread = Threads.nthreads()
println("I am using $nthread threads")

function custom_square(arr)
    
end

arr = rand(1000)
println(custom_square(arr))
    
@btime custom_square(arr)