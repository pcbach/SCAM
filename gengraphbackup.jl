using LinearAlgebra
using Distributions
using Plots
using BenchmarkTools
using Statistics
using Arpack
using Graphs
using GraphPlot
using Compose
using Cairo
using Fontconfig
using DelimitedFiles

function disp(a)
    display(a)
    print('\n')
end

i = Int64(readdlm("Count.csv", ',', Float64, '\n')[1])
para = readdlm("parameter.csv", ',', Float64, '\n')
n = 50
m = 49
C = zeros(n, n)
A = zeros(m, n)
g = Graph(n)
#=
j = 2
A[1,1] = 1
A[1,2] = -1
C[1,2] = 1
C[2,1] = 1
add_edge!(g, 1, 2)
=#
for i in 1:n-1
    j = rand(i+1:n)
    A[i, i] = 1
    A[i, j] = -1
    C[i, j] = 1
    C[j, i] = 1
    add_edge!(g, i, j)
end

for i in n:m
    print(i, '-')
    j = rand(1:n-1)
    k = rand(j+1:n)
    while C[j, k] == 1
        j = rand(1:n-1)
        k = rand(j+1:n)
    end
    A[i, j] = 1
    A[i, k] = -1
    C[j, k] = 1
    C[k, j] = 1
    add_edge!(g, k, j)
end

print(n, "-", m)
layout = (args...) -> spring_layout(args...; C=10)
draw(PNG("graphbackup.png", 35cm, 35cm), gplot(g, layout=random_layout, nodelabel=1:n, nodesize=5, nodelabelsize=10))
#disp(A)
writedlm("graphbackup.csv", A, ',')
#disp(C)
#A = readdlm("graph.csv", ',', Float64, '\n')
#disp(A)