include("MESDP.jl")
include("ReadGSet.jl")
using Plots

colorCount = 1
Result = zeros((29, 2))
function exp1(inputFile, optimalValue, label; linesearch=false, ε=1e-2, v0=nothing, t0=0, new=false)
    file = inputFile
    A = readfile(file)
    global m = size(A, 1)
    global n = size(A, 2)
    A = A / 2
    #evareC = A * A'
    A_s = sparse(A)
    A = nothing

    opt = sqrt(optimalValue * 2)
    #Calculate graph degree
    if new
        rows = rowvals(A_s)
        vals = nonzeros(A_s)
        D = spzeros(n)
        for i in 1:n
            for k in nzrange(A_s, i)#
                D[i] += vals[k]^2
            end
        end
        lower = 0
        upper = sum(D)
    else
        rows = rowvals(A_s)
        vals = nonzeros(A_s)
        D = spzeros(n)
        for i in 1:n
            for k in nzrange(A_s, i)#
                D[i] += 1
            end
        end
        lower = opt
        upper = 2 * opt
    end
    rows = nothing
    vals = nothing

    v = B(A_s, identity=1 / m)
    p = nothing
    t = 2
    if v0 !== nothing
        v = v0
        t = t0
    end
    result1 = Solve(A_s, v, t0=t, D=D, lowerBound=lower, upperBound=upper, printIter=true, plot=true, linesearch=linesearch, ε=ε, numSample=1)
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
    cv = CutValue(A_s, r)
    #disp(Vector(cv.cut), name="Cut")
    disp(result1.val, name="Primal objective")
    disp(cv.val, name="Cut Value")
    Result[colorCount, 1] = CutValue(A_s, r).val / 2
    Result[colorCount, 2] = result1.val^2 / 2
    global colorCount = colorCount + 1
    A_s = nothing
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
idx = ["1", "14", "22"]
for row in csv_reader
    #println(row.filename, " ", row.opt)
    index = chop(row.filename, head=3, tail=0)
    if index in idx
        opt = row.opt
        result = exp1("Gset/g" * index * ".txt", -opt,
            "G" * index, ε=ε1, linesearch=false)

        result = exp1("Gset/g" * index * ".txt", -opt,
            "G" * index * "new", ε=ε1, linesearch=false, new=true)
    end
end

savefig("Result/exp2/Gset1-14-22Cheat.png")
#CSV.write("ResultMESDP.csv", Tables.table(Result), writeheader=false)