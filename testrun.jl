include("MESDP.jl")
include("genGraph.jl")
include("readdata.jl")
using BenchmarkTools

using Plots


A = readfile("Gset/g35.txt")
disp(A)
println("done")
global m = size(A, 1)
global n = size(A, 2)
A = A / 2
A_s = sparse(A)

p = ones((n, 1))
p[1] = -1
p = p / norm(p)
#p = rand(Uniform(-1, 1), (n, 1))
p = A * p
v = B(A, p)

println("start")


result = Solve(A_s, v, lowerBound=0, upperBound=1e16, printIter=true, plot=true, ε=1e-2)
disp(result.val^2 / 2)
#plot!(log10.(1:length(result.plot)), log10.((sqrt(n_ - 1) .- result.plot) ./ sqrt(n_ - 1)), ratio=:equal)

#result = Solve(A, v, lowerBound=1, upperBound=1e16, printIter=true, plot=true, ε=7e-3)
#plot!(log10.(1:length(result.plot)), log10.((sqrt(499) .- result.plot) ./ sqrt(499)), ratio=:equal)
#writedlm("log500_2.csv", result.plot, ',')
#plot(log10.(1:length(result.plot)), log10.((sqrt(10000) .- result.plot) ./ sqrt(10000)), ratio=:equal)
#savefig("200/f2.png")

plot(diff(result.time))
disp(result.time[length(result.time)] - result.time[1])
savefig("time.png")
