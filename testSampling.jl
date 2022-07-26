include("MESDP.jl")
include("ReadGSet.jl")
using Plots


colorCount = 2
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
    rows = nothing
    vals = nothing

    v = B(A_s, P=diagm(ones(m)) / m)
    disp(f(A_s, v), name="f(x)")
    p = nothing
    t = 2
    opt = sqrt(optimalValue * 2)
    if v0 !== nothing
        v = v0
        t = t0
    end

    #=
    for i in 1:20
        z = rand(Normal(0, sqrt(1 / m)), (1, m))
        disp(CutValue(A_s, z[1, :]) / 2)
    end
    =#
    result1 = Solve(A_s, v, t0=t, D=D, lowerBound=opt, upperBound=opt * 2, printIter=true, plot=true, linesearch=linesearch, ε=ε, numSample=1)
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
    plotx = t0 - 1 .+ (1:length(result1.plot.y))
    #ploty = log10.(opt .- (result1.plot))
    ploty = (opt .- result1.plot.y) ./ opt
    #ploty = result1.plot.y
    #disp(plotx)
    #disp(ploty)
    if linesearch
        style = :dash
    else
        style = :solid
    end
    plot!(log10.(plotx), log10.(ploty), label=label, dpi=300, size=(1000, 1000), color=Int64(floor(colorCount / 2)),
        lw=3, legend_font_pointsize=20, tickfontsize=20, legend_position=:bottomleft, style=style)
    #=plotyTheo = log10.(((32 * sum(D) / minimum(D)) ./ (plotx)))
    plot!(log10.(plotx), plotyTheo, label="", dpi=300, size=(1000, 1000), color=colorCount,
        lw=3, style=:dash, legend_font_pointsize=20, tickfontsize=20, legend_position=:bottomleft)


    =#
    global colorCount = colorCount + 1
    if outputFile !== nothing
        savefig(outputFile)
    end
    r = result1.z
    disp(CutValue(A_s, r) / 2, name="Cut Value")
    disp(result1.val^2 / 2, name="Primal objective")
    #disp(result1.val)
    return (v=result1.v, t=result1.t, z=result1.z)
end
#"C:/Users/pchib/Desktop/MASTER/MESDP/toy.txt"
#print(m,n)
ε1 = 10^(-2)
ε2 = 1e-3
pyplot();

opt = 3191.566782866967
inputfile = "Gset/g14.txt"
result = exp1(inputfile, nothing, opt,
    "G14ls", ε=ε1, linesearch=true)
opt = 3191.566782866967
inputfile = "Gset/g14.txt"
result = exp1(inputfile, nothing, opt,
    "G14", ε=ε1, linesearch=false)


opt = 6000
inputfile = "Gset/g48.txt"
result = exp1(inputfile, nothing, opt,
    "G48", ε=ε1, linesearch=true)
opt = 6000
inputfile = "Gset/g48.txt"
result = exp1(inputfile, nothing, opt,
    "G48", ε=ε1, linesearch=false)


opt = 8018.623274609614
inputfile = "Gset/g37.txt"
result = exp1(inputfile, nothing, opt,
    "G37ls", ε=ε1, linesearch=true)
opt = 8018.623274609614
inputfile = "Gset/g37.txt"
result = exp1(inputfile, nothing, opt,
    "G37", ε=ε1, linesearch=false)


opt = 14135.94557910069
inputfile = "Gset/g22.txt"
result = exp1(inputfile, nothing, opt,
    "G22ls", ε=ε1, linesearch=true)
opt = 14135.94557910069
inputfile = "Gset/g22.txt"
result = exp1(inputfile, "Result/exp1/G14-G48-G37-G22_low.png", opt,
    "G22", ε=ε1, linesearch=false)

