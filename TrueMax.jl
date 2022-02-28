using Convex, Mosek, MosekTools, JuMP
using DelimitedFiles
using BenchmarkTools

#===========================GLOBAL VAR==========================#

m = 0
n = 0

#==============================MISC=============================#

function disp(quan;name = "")
    if name != ""
        print(name,":\n")
    end
    display(quan)
    print('\n')
end

#=============================SOLVE=============================#

function CVXsolve(A)
    v = []
    tau = []
    P = Variable(m, m)
    constraints = (P ⪰ 0)
    constraints += (tr(P) == 1)

    for i in 1:n
        append!(v,Variable())
        append!(tau,Variable())
    end

    for i in 1:n
        ai = vec(A[:,i])
        constraints += (v[i] >= 0)
        constraints += (tau[i] >= 0)
        constraints += (v[i] == ai' * P * ai)
        constraints += ([v[i] tau[i];tau[i] 1] ⪰ 0)
    end

    problem = maximize(sum(tau),constraints)
    solve!(problem, Mosek.Optimizer, verbose=true)

    #disp(problem.optval^2)
    return problem.optval^2
end

#==============================MAIN=============================#

function main()
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A/2
    global m = size(A,1)
    global n = size(A,2)
    
    writedlm("result.csv", CVXsolve(A), ',')
end 

main()