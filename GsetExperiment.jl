include("MESDP.jl")
include("ReadGSet.jl")
using Plots

colorCount = 0
Result = zeros((29, 3))
function exp1(inputFile, optimalValue, label; linesearch=false, ε=1e-2, v0=nothing, t0=0, mode="new")
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
    if mode == "new"
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
    elseif mode == "opt"
        rows = rowvals(A_s)
        vals = nonzeros(A_s)
        D = spzeros(n)
        for i in 1:n
            for k in nzrange(A_s, i)#
                D[i] += 1
            end
        end
        lower = 0
        upper = 2 * opt
    elseif mode == "none"
        D = ones(n)
        lower = 0
        upper = 1e16
    end

    #println(maximum(upper ./ D), " ", minimum(upper ./ D), " ", mean(upper ./ D), " ", median(upper ./ D))

    rows = nothing
    vals = nothing

    v = B(A_s, identity=1 / m)
    p = nothing
    t = 2
    if v0 !== nothing
        v = v0
        t = t0
    end
    disp(upper)
    disp(lower)
    result1 = Solve(A_s, v, t0=t, D=D, lowerBound=lower, upperBound=upper, printIter=true, plot=true, linesearch=linesearch, ε=ε, numSample=20)
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
    global colorCount = colorCount + 1
    Result[colorCount, 1] = CutValue(A_s, r).val / 2
    Result[colorCount, 2] = result1.val^2 / 2
    Result[colorCount, 3] = result1.t
    A_s = nothing
    #histogram(∇g(v))
    disp(Result)
    return (v=result1.v, t=result1.t, z=result1.z)

end

ε1 = 1 * 10^(-2)
pyplot();
#=

opt = 3191.566782866967
inputfile = "Gset/g14.txt"

savefig("Result/exp2/Gset.png")
=#

using CSV, Tables
csv_reader = CSV.File("result.csv")
idx = string.([collect(1:5); collect(14:17); collect(22:26); collect(35:37); collect(43:54)]);
for row in csv_reader
    #println(row.filename, " ", row.opt)
    index = chop(row.filename, head=3, tail=0)
    if index in idx
        opt = row.opt
        result = exp1("Gset/g" * index * ".txt", -opt,
            "G" * index, ε=ε1, linesearch=false, mode="new")
        result = exp1("Gset/g" * index * ".txt", -opt,
            "G" * index, ε=ε1, linesearch=false, mode="new")
    end
end
disp(Result)
savefig("Result/exp2/Gsetnew.png")
CSV.write("ResultMESDP.csv", Tables.table(Result), writeheader=false)