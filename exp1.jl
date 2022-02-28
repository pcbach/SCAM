include("./Samp.jl")
using LinearAlgebra
using Distributions
using Plots
using BenchmarkTools
using Statistics
using Arpack 
using DelimitedFiles

function genSample(m)
    p = rand(Uniform(-1, 1),(m,1))
    X0 = p*p'
    val,vec = eigen(X0)
    val = max.(1e-5,val)
    X0 = Matrix(Hermitian(vec*Diagonal(val)*inv(vec)))
    z = rand(MvNormal(zeros(m),X0))
    return z
end

A = readdlm("graph.csv", ',', Float64, '\n')
m = size(A,1)
n = size(A,2)

#=
z = genSample(m)
print("\nTrue\n")
display(solvesampTrue(A,z))
print("\nPow\n")
display(solvesampPow(A,z))
print("\n")
=#

#display(A)
z = genSample(m)
print("\nTrue\n")
display(@benchmark solvesampTrue(A,z))
print("\nPow\n")
display(@benchmark solvesampPow(A,z,1e-3))
print("\n")


maxepoch = 100
maxrep = 1
tally1 = zeros(0)
tally2 = zeros(0)
for epoch in 1:maxepoch
    print("\n",epoch,"\n")
    result1 = 0
    result2 = 0
    for rep in 1:maxrep
        z = genSample(m)
        re1 = solvesampTrue(A,z)[1]
        re2 = solvesampPow(A,z,1e-2)[1]
        result1 = max(re1, result1) 
        result2 = max(re2, result2)
        
    end
    append!(tally1,result1) 
    append!(tally2,result2) 
end

mvavg1 = zeros(maxepoch)
mvavg2 = zeros(maxepoch)
cumsum1 = 0
cumsum2 = 0
for i in 1:maxepoch
    global cumsum1 = cumsum1 + tally1[i]
    mvavg1[i] = cumsum1/i
    global cumsum2 = cumsum2 + tally2[i]
    mvavg2[i] = cumsum2/i
end

gr()
plot(mvavg1)
plot!(mvavg2)
print(maximum(tally1),'\n')
print(maximum(tally2),'\n')
print(mvavg1[maxepoch],'\n',mvavg2[maxepoch],'\n')
savefig("plotexp2.png")
