using Convex
using MosekTools
using LinearAlgebra
using Statistics
include("ReadGSet.jl")


A = readfile("Gset/g1.txt")
m = size(A, 1)
n = size(A, 2)
function solve1()
    u = Variable(5)
    problem = minimize(sum(u))
    I = diagm(ones(7))
    k = vcat(hcat(diagm(u), A'), hcat(A, I))
    problem.constraints += [k ⪰ 0]
    # Solve the problem by calling solve!
    solve!(problem, Mosek.Optimizer; silent_solver=true)

    # Check the status of the problem
    println(problem.status) # :Optimal, :Infeasible, :Unbounded etc.

    # Get the optimum value
    println(problem.optval)
    println(evaluate(u))
end

function solve2()
    ζ = Variable(n)
    δ = Variable(1)
    problem = minimize(sum(ζ) + δ)
    ω = Variable(n)
    k = Variable(m, m)
    I = diagm(ones(m))
    problem.constraints += [δ >= 0]
    problem.constraints += [k >= 0]
    for i in 1:n
        a = A[:, i]
        problem.constraints += [ω[i] >= 0]
        problem.constraints += [[ω[i] 0.5; 0.5 ζ[i]] ⪰ 0]
        k += ω[i] * a * a'
    end

    problem.constraints += [k ⪯ δ * I]
    # Solve the problem by calling solve!
    solve!(problem, Mosek.Optimizer; silent_solver=true)

    # Check the status of the problem
    println(problem.status) # :Optimal, :Infeasible, :Unbounded etc.

    # Get the optimum value
    return problem.optval
end

#solve1()
#opt = sqrt(12083.2 * 2)
#display(opt)

#Calculate graph degree
#=D = zeros(n)
for i in 1:n
    for j in 1:m
        D[i] += abs(A[j, i])
    end
end

println(sqrt(sum(D) / 2) / (2 * opt))
=#
using CSV, Tables
function ratio()
    csv_reader = CSV.File("result.csv")
    for row in csv_reader
        #println(row.filename, " ", row.opt)
        index = chop(row.filename, head=3, tail=0)
        opt = -row.opt
        opt = sqrt(opt * 2)
        A = readfile("Gset/g" * index * ".txt")
        m = size(A, 1)
        n = size(A, 2)
        A = A / 2
        D = zeros(n)
        for i in 1:n
            a = A[:, i]
            D[i] = norm(a)^2
        end
        println("G" * index * ": ", opt^2 / 2, " ", sum(D), " ")
    end
end
function testFeasibility()
    ω = ones(n) * opt ./ (D)
    k = zeros(m, m)
    for i in 1:n
        println(i)
        a = A[:, i]
        #println(maximum(eigen(a * a').values))
        k += ω[i] * a * a'
        #println(maximum(eigen(k).values))
    end
    display(maximum(eigen(k).values))
end

#testFeasibility()

#ratio()

D = Variable(15)
s = Variable()

problem = maximize(D[1])

problem.constraints += [s > 0.4]
problem.constraints += [D[1] + s < D[2]]
problem.constraints += [D[2] == D[3]]
problem.constraints += [D[2] == D[4]]
problem.constraints += [D[2] == D[5]]
problem.constraints += [D[2] == D[6]]
problem.constraints += [D[2] == D[7]]
problem.constraints += [D[2] == D[8]]
problem.constraints += [D[8] + s < D[9]]
problem.constraints += [D[9] == D[10]]
problem.constraints += [D[9] == D[11]]
problem.constraints += [D[9] == D[12]]
problem.constraints += [D[12] + s < D[13]]
problem.constraints += [D[13] + s < D[14]]
problem.constraints += [D[14] + s < D[15]]
problem.constraints += [sum(D) == 42]
solve!(problem, Mosek.Optimizer; silent_solver=true)
print(evaluate(D))
