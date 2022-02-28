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

n = 10
m = 20
C = zeros(n,n)
A = zeros(m,n)
g = Graph(n)

for i in 1:n-1
    j = rand(i+1:n)
    A[i,i] = 1
    A[i,j] = -1
    C[i,j] = 1
    C[j,i] = 1
    add_edge!(g, i, j)
end

for i in n:m
    j = rand(1:n-1)
    k = rand(j+1:n)
    while C[j,k] == 1
        j = rand(1:n-1)
        k = rand(j+1:n)
    end
    A[i,j] = 1
    A[i,k] = -1
    C[j,k] = 1
    C[k,j] = 1
    add_edge!(g, k, j)
end


#layout=(args...)->spring_layout(args...; C=30)
#draw(PNG("graph.png", 35cm, 35cm), gplot(g, layout = layout, nodelabel = 1:n,nodesize = 5, nodelabelsize = 10, linetype="curve"))
#disp(A)
writedlm("graph.csv",  A, ',')
#disp(C)
#A = readdlm("graph.csv", ',', Float64, '\n')
#disp(A)