include("MESDP.jl")
include("ReadGSet.jl")
using LaTeXStrings
using Plots
using Plots.PlotMeasures
using Plots
using BenchmarkTools


colorCount = 2
#Single graph with individual bound on the gradient
function exp1(inputFile, label; linesearch=false, ε=1e-2, v0=nothing, t0=0, bound=true, color=:black, mode="A")
    file = inputFile
    disp(file, name="file")
    @time A, C = readfile(file)
    #disp(size(A))
    A = A / 2
    C = C / 4
    #disp(norm(C - A' * A), name="norm")
    global m = size(A, 1)
    global n = size(A, 2)

    sumai = 0
    D = zeros(n)
    for i = 1:n
        sumai += 2 * C[i, i]
        D[i] = 2 * C[i, i]
    end
    sumai = sqrt(sumai)

    #disp(sumai, name="SUMAI")
    #disp(D, name="DD")
    #disp(minimum(D))
    rows = nothing
    vals = nothing
    A_s = sparse(A)
    v = B(A_s, d=1 / m)
    #print(f(A_s, v), " ", n, " ", m, " ")

    t = 5
    if v0 !== nothing
        v = v0
        t = t0
    end
    if bound
        upper = sumai
    else
        D = ones(n)
        upper = 1e16
    end
    if mode == "A"
        result1 = Solve(A_s, v, t0=t, D=D, lowerBound=0, upperBound=upper, printIter=true, plot=true, linesearch=linesearch, ε=ε, numSample=1, mode=mode)
    elseif mode == "C"
        result1 = Solve(C, v, t0=t, D=D, lowerBound=0, upperBound=upper, printIter=true, plot=true, linesearch=linesearch, ε=ε, numSample=1, mode=mode)
    end

    style = :solid
    #disp(log10.((opt .- result1.plot) ./ opt))

    plot!(log10.(t0 - 1 .+ (1:length(result1.plot))), log10.(result1.gap), ratio=:equal, label=label, dpi=600, size=(150, 150),
        lw=0.5, lc=color, gridalpha=0.5, ls=style, legend_font_pointsize=4, tickfontsize=5, legend_position=:bottomleft, xlabel=L"\ln(k)", ylabel=L"\ln(RFWgap)", yguidefontsize=5, xguidefontsize=5, left_margin=-0.5mm, bottom_margin=-0.5mm, ytick=collect(-5:1:5), xtick=collect(-5:1:5))

    open("Result/Log/exp5/" * chop(inputFile, head=11, tail=4) * "log.txt", "w") do io
        println(io, Int64(length(result1.gap)))
        for i in 1:length(result1.gap)
            println(io, result1.gap[i])
        end
    end
    #if outputFile !== nothing
    #    savefig(outputFile)
    #end
    r = result1.z
    disp(result1.val^2 / 2)
    disp(result1.t)
    return (v=result1.v, t=result1.t, z=result1.z)
end
#=
inputfile = "toy.txt"
@time exp1(inputfile, nothing, ε=10^(-2), linesearch=true, bound=true, color=:black, mode="C")
=#
#opt = 12083.2
#ε1 = 10^(-2.5)
#inputfile = "Gset/g1.txt"
#@time exp1(inputfile, nothing, ε=ε1, linesearch=true, bound=true, color=:black, mode="A")
#@time exp1(inputfile, nothing, ε=ε1, linesearch=true, bound=true, color=:black, mode="C")
#=
@time A, C = readfile("BigExample/100kn3d.txt")
global n = size(C, 1)
global m = size(A, 1)
v = B(A, d=1 / m)
@time a1, b1, c1 = ArnoldiGrad(A, v, D=ones(n), mode="A")
@time a2, b2, c2 = ArnoldiGrad(C, v, D=ones(n), mode="C")
disp(norm(b1 - b2))
=#
#=
ε1 = 10^(-2.5)
num = [1, 2, 3, 4, 5,
    14, 15, 16, 17, 22,
    23, 24, 25, 26, 35,
    36, 37, 43, 44, 45,
    46, 47, 48, 49, 50,
    51, 52, 53, 54];
optval = [12083.2, 12089.43, 12084.33, 12111.45, 12099.89,
    3191.57, 3171.56, 3175.02, 3171.33, 14135.95,
    14142.22, 14140.86, 14144.25, 14132.87, 8014.74,
    8005.96, 8018.62, 7032.22, 7027.88, 7024.78,
    7029.93, 7036.66, 6000, 6000, 5988.17,
    4006.26, 4009.64, 4009.72, 4006.19];
ε = [0.015, 0.017, 0.015, 0.014, 0.013,
    0.05, 0.048, 0.04, 0.044, 0.013,
    0.015, 0.015, 0.014, 0.015, 0.033,
    0.044, 0.044, 0.015, 0.014, 0.017,
    0.015, 0.015, 0.009, 0.01, 0.007,
    0.035, 0.034, 0.043, 0.039]
label = ["Group1", "Group2", "Group3", "Group4", "Group5", "Group6", "Group7"];
labeli = [1, 6, 10, 15, 18, 23, 26];
outputfile = "Result/exp1/test.png"
colors = [:pink, :red, :yellow, :orange, :blue, :black, :cyan]
global colorcnt = 0
cnt = 29
for i in 1:cnt
    inputfile = "Gset/g" * string(num[i]) * ".txt"
    #inputfile = "C:/Users/pchib/Desktop/MASTER/MESDP/Gset/g3k" * string(num[i]) * ".txt"
    opt = optval[i]
    print(string(num[i]) * ": ")
    if i in labeli
        global colorcnt = colorcnt + 1
        idx = findall(x -> x == i, labeli)
        lbl = label[idx]
        result1 = exp1(inputfile, nothing, opt,
            lbl[1], ε=ε[i] / sqrt(2), linesearch=true, bound=true, color=colors[colorcnt], mode="C")
    else
        lbl = nothing
        result1 = exp1(inputfile, nothing, opt,
            nothing, ε=ε[i] / sqrt(2), linesearch=true, bound=true, color=colors[colorcnt], mode="C")
    end

end

savefig(outputfile)
=#


inputfile = ["100kn3d.txt", "100kn3d2.txt", "100kn10d.txt", "100kn10d2.txt", "300kn3d.txt", "300kn3d2.txt", "300kn5d.txt", "300kn5d.txt", "1mn3d.txt", "1mn3d2.txt", "1mn4d.txt", "1mn4d2.txt"]

for i in 1:length(inputfile)
    name = inputfile[i]
    file = "BigExample/" * name
    exp1(file, name, ε=10^(-2), linesearch=true, bound=true, mode="A")
end
