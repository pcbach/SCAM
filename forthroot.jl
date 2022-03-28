using Convex, Mosek, MosekTools, JuMP
using DelimitedFiles
using BenchmarkTools
using LinearAlgebra

#===========================GLOBAL VAR==========================#

m = 0
n = 0

#==============================MISC=============================#

function disp(quan; name="")
    if name != ""
        print(name, ":\n")
    end
    display(quan)
    print('\n')
end

#=============================SOLVE=============================#

function CVXsolve(A)
    v = Variable(n)
    τ = Variable(n)
    ψ = Variable(n)
    P = Variable(m, m)
    constraints = (P ⪰ 0)
    constraints += (tr(P) == 1)


    for i in 1:n
        ai = vec(A[:, i])
        constraints += (v[i] >= 0)
        constraints += (τ[i] >= 0)
        constraints += (v[i] == ai' * P * ai)
        constraints += ([v[i] τ[i]; τ[i] 1] ⪰ 0)
        #constraints += ([τ[i] ψ[i];ψ[i] 1] ⪰ 0)
    end

    problem = maximize(sum(τ), constraints)
    solve!(problem, Mosek.Optimizer, verbose=false)
    println(evaluate(v))
    return problem.optval
end

#==============================MAIN=============================#
function f_(A, P)
    n = size(A, 2)
    r = 0
    for i in 1:n
        a = A[:, i]
        r = r + sqrt(a' * P * a)
    end
    return r
end

function main()
    A = readdlm("graphbackup.csv", ',', Float64, '\n')
    A = A / 2
    global m = size(A, 1)
    global n = size(A, 2)
    #disp(f_(A,w*w'))
    #disp(A'*A)
    disp(CVXsolve(A))
    #writedlm("result.csv", CVXsolve(A), ',')
end

main()