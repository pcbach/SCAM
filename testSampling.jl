include("MESDP.jl")
include("ReadGSet.jl")
using Arpack
using Plots


colorCount = 2
#Single graph with individual bound on the gradient
function exp1(inputFile, outputFile, optimalValue, label; linesearch=false, ε=1e-2, v0=nothing, t0=0, D_sp=0)
    file = inputFile
    A = readfile(file)
    global m = size(A, 1)
    global n = size(A, 2)
    A = A / 2
    #C = A * A'
    A_s = sparse(A)
    #=
    D_s = spzeros(n, n)
    for i in 1:n
        D_s[i, i] = D_sp
    end
    A_s = vcat(A, D_s)
    global m = size(A_s, 1)
    global n = size(A_s, 2)
    disp(A_s)
    =#
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

    v = B(A_s, identity=1 / (m))
    #disp(f(A_s, v), name="f(x)")
    p = nothing
    t = 2
    opt = sqrt(optimalValue * 2)
    if v0 !== nothing
        v = v0
        t = t0
    end
    rows = rowvals(A_s)
    vals = nonzeros(A_s)
    D = spzeros(n)
    for i in 1:n
        for k in nzrange(A_s, i)#
            D[i] += vals[k]^2
        end
    end
    lower = 0
    upper = sqrt(2 * sum(D))
    disp(sum(D) / minimum(D), name="ratio")
    result1 = SolveSketch(A_s, v, t0=t, D=D, lowerBound=lower, upperBound=upper, printIter=true, plot=true, linesearch=linesearch, ε=ε)
    plotx = t0 - 1 .+ (1:length(result1.plot.y))
    ploty = result1.plot.y
    if linesearch
        style = :dash
    else
        style = :solid
    end
    plot!(log10.(plotx), ploty, label=label, dpi=300, size=(1000, 1000), color=Int64(floor(colorCount)),
        lw=3, legend_font_pointsize=20, tickfontsize=20, legend_position=:bottomright, style=style)
    global colorCount = colorCount + 1
    if outputFile !== nothing
        savefig(outputFile)
    end
    #r = result1.z
    #disp(CutValue(A_s, r).val / 2, name="Cut Value")
    #disp(result1.val^2 / 2, name="Primal objective")
    #disp(result1.val)
    return (v=result1.v, t=result1.t)#, z=result1.z)
end
#"C:/Users/pchib/Desktop/MASTER/MESDP/toy.txt"
#print(m,n)
ε1 = 10^(-1)
ε2 = 1e-3
pyplot();

opt = 6000
inputfile = "toy.txt"

result = exp1(inputfile, nothing, opt,
    "G14", ε=ε2, linesearch=false)

savefig("Result/toy.png")