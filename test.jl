include("MESDP.jl")
include("ReadGSet.jl")

using Plots
#Single graph with individual bound on the gradient
function exp1(inputFile, outputFile, optimalValue, option; ε=1e-2)
    file = inputFile
    A = readfile(file)
    global m = size(A, 1)
    global n = size(A, 2)
    A = A / 2
    A_s = sparse(A)
    A = nothing

    #Calculate graph degree
    rows = rowvals(A_s)
    vals = nonzeros(A_s)
    D = spzeros(n)
    for i in 1:n
        for k in nzrange(A_s, i)#
            D[i] += 1
        end
    end
    rows = nothing
    vals = nothing

    p = ones((n, 1))
    if option == "random"
        #p[rand(1:n)] = -1
        p = rand(Uniform(-1, 1), (n, 1))
    elseif option == "one"
        p[1] = -1
    elseif option == "min"
        p[argmin(D)] = -1
    elseif option == "max"
        p[argmax(D)] = -1
    end

    p = A_s * p
    p = p / norm(p)
    v = B(A_s, v=p)
    p = nothing

    opt = sqrt(optimalValue * 2)
    result1 = Solve(A_s, v, D=D, lowerBound=opt, upperBound=opt * 2, printIter=true, plot=true, linesearch=false, ε=ε)
    disp(result1.val^2)
    plot!(log10.((1:length(result1.plot))), log10.((opt .- result1.plot) ./ opt), ratio=:equal, label=option, dpi=300, size=(1000, 1000), lw=3)
    if outputFile !== nothing
        savefig(outputFile)
    end
end

exp1("Gset/G5.txt", nothing, 12100, "min")
exp1("Gset/G14.txt", "Result/exp1/G14and5.png", 3191.6, "max")