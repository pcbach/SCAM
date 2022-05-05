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
function genGraph(n, m)
    C = spzeros(n, n)
    A = spzeros(m, n)

    for i in 1:n-1
        j = rand(i+1:n)
        A[i, i] = 1
        A[i, j] = -1
        C[i, j] = 1
        C[j, i] = 1
    end

    for i in n:m
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
    end

    A = A / 2
    return A
end