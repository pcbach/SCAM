include("MESDP.jl")
include("ReadGSet.jl")

using Plots
#Single graph with individual bound on the gradient
function exp1(inputFile, outputFile, optimalValue, label; linesearch=false, ε=1e-2, v0=nothing, t0=0)
    file = inputFile
    A = readfile(file)
    global m = size(A, 1)
    global n = size(A, 2)
    A = A / 2
    C = A * A'
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
    disp(sum(D) / minimum(D))
    rows = nothing
    vals = nothing

    v = B(A_s, P=diagm(ones(m)) / m)
    #disp(v)
    p = nothing
    t = 2
    opt = sqrt(optimalValue * 2)
    if v0 !== nothing
        v = v0
        t = t0
    end
    result1 = Solve(A_s, v, t0=t, D=D, lowerBound=opt, upperBound=opt * 2, printIter=true, plot=true, linesearch=linesearch, ε=ε, numSample=10)
    ########################################################
    #=P = Badj(A_s,result1.v)
    disp(P)
    U,s,V = svd(Matrix(P))
    ω = U[:,1]*sqrt(s[1])
    x = sign.(pinv(Matrix(A_s))*ω)
    x = x/norm(x)
    v = B(A_s,v = x)
    disp(f(A_s,v))=#
    #########################################################
    #disp(result1.val^2 / 2)
    plot!(log10.(t0 - 1 .+ (1:length(result1.plot))), log10.((opt .- result1.plot) ./ opt), ratio=:equal, label=label, dpi=300, size=(1000, 1000),
        lw=3, legend_font_pointsize=20, tickfontsize=20, legend_position=:bottomleft)
    if outputFile !== nothing
        savefig(outputFile)
    end
    r = result1.z
    disp(CutValue(A_s, r) / 2)
    #disp(result1.val)
    return (v=result1.v, t=result1.t, z=result1.z)
end
#"C:/Users/pchib/Desktop/MASTER/MESDP/toy.txt"
#print(m,n)
ε1 = 1e-2
ε2 = 1e-3

opt = 12083
inputfile = "Gset/g1.txt"
result = exp1(inputfile, nothing, opt,
    "G1", ε=3e-2, linesearch=true)

opt = 3191.6
inputfile = "Gset/g14.txt"
result = exp1(inputfile, nothing, opt,
    "G14", ε=3e-2, linesearch=true)

opt = 7032.2
inputfile = "Gset/g43.txt"
result = exp1(inputfile, nothing, opt,
    "G43", ε=3e-2, linesearch=true)

opt = 6000
inputfile = "Gset/g48.txt"
result = exp1(inputfile, nothing, opt,
    "G48", ε=3e-2, linesearch=true)

opt = 14136
inputfile = "Gset/g22.txt"
result = exp1(inputfile, "Result/exp1/G1-14-22.png", opt,
    "G22", ε=3e-2, linesearch=true)
#disp(result.z)

