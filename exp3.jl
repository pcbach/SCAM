include("MESDP.jl")
include("ReadGSet.jl")
using Plots

colorCount = 1
Result = zeros((29, 3))
function exp1(inputFile, optimalValue, label; linesearch=false, ε=1e-2, v0=nothing, t0=0)
    file = inputFile
    A = readfile(file)
    global m = size(A, 1)
    global n = size(A, 2)
    A = A / 2
    #C = A * A'
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

    v = B(A_s, identity=1 / m)
    p = nothing
    t = 2
    opt = sqrt(optimalValue * 2)
    if v0 !== nothing
        v = v0
        t = t0
    end

    result1 = FrankWolfeSolve(A_s, v, t0=t, D=D, lowerBound=opt, upperBound=opt * 2, printIter=true, plot=true, linesearch=linesearch, ε=ε, numSample=1)
    plotx = t0 - 1 .+ (1:length(result1.plot.y))
    ploty = (opt .- result1.plot.y) ./ opt
    if n == 800
        style = :dash
    elseif n == 1000
        style = :solid
    elseif n == 2000
        style = :dashdot
    elseif n == 3000
        style = :dot
    end
    plot!(log10.(plotx), log10.(ploty), label=label, dpi=300, size=(1000, 1000), color=colorCount,
        lw=3, legend_font_pointsize=10, tickfontsize=20, legend_position=:bottomleft, style=style)

    r = result1.z
    disp(label)
    disp(CutValue(A_s, r) / 2, name="Cut Value")
    disp(result1.val^2 / 2, name="Primal objective")
    disp(result1.t, name="Iteration")
    Result[colorCount, 1] = CutValue(A_s, r) / 2
    Result[colorCount, 2] = result1.val^2 / 2
    Result[colorCount, 3] = result1.t
    global colorCount = colorCount + 1

    result1 = FrankWolfeSolve_test(A_s, v, t0=t, D=D, lowerBound=opt, upperBound=opt * 2, printIter=true, plot=true, linesearch=linesearch, ε=ε * opt, numSample=1)
    plotx = t0 - 1 .+ (1:length(result1.plot.y))
    ploty = (opt .- exp.(result1.plot.y)) ./ opt
    if n == 800
        style = :dash
    elseif n == 1000
        style = :solid
    elseif n == 2000
        style = :dashdot
    elseif n == 3000
        style = :dot
    end
    plot!(log10.(plotx), log10.(ploty), label=label * "log", dpi=300, size=(1000, 1000), color=colorCount,
        lw=3, legend_font_pointsize=10, tickfontsize=20, legend_position=:bottomleft, style=style)

    r = result1.z
    disp(label)
    disp(CutValue(A_s, r) / 2, name="Cut Value")
    disp(exp(result1.val * 2) / 2, name="Primal objective")
    disp(result1.t, name="Iteration")
    Result[colorCount, 1] = CutValue(A_s, r) / 2
    Result[colorCount, 2] = exp(result1.val * 2) / 2
    Result[colorCount, 3] = result1.t
    global colorCount = colorCount + 1
    return (v=result1.v, t=result1.t, z=result1.z)
end

ε1 = 10^(-2)
pyplot();
#=

opt = 3191.566782866967
inputfile = "Gset/g14.txt"

savefig("Result/exp2/Gset.png")
=#

using CSV, Tables
csv_reader = CSV.File("result.csv")
println("Read file")
idx = ["14"]
for row in csv_reader
    #println(row.filename, " ", row.opt)
    index = chop(row.filename, head=3, tail=0)
    if (index in idx)
        println("G", index)
        opt = row.opt
        result = exp1("Gset/g" * index * ".txt", -opt,
            "G" * index, ε=ε1, linesearch=false)
    end
end

savefig("Result/exp3/GsetHigh.png")
disp(Result)
#CSV.write("ResultMESDP_2.csv", Tables.table(Result), writeheader=false)