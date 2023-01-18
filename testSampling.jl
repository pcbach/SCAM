include("MESDP.jl")
include("ReadGSet.jl")
using LaTeXStrings
using Plots
using Plots.PlotMeasures
using BenchmarkTools

#Single graph with individual bound on the gradient
function exp1(inputFile, outputfile; linesearch=false, ε=1e-2, v0=nothing, t0=0, bound=true, mode="A", startεd0=0.0)
    file = inputFile
    A, C = readfile(file)
    #disp(size(A))
    A = A / 2
    C = C / 4
    global m = size(A, 1)
    global n = size(A, 2)

    D = spzeros(n)
    sumai = 0
    for i in 1:n
        D[i] = 2 * C[i, i]
        sumai = sumai + D[i]
    end
    disp(sumai)
    sumai = sqrt(sumai)
    v = B(A, d=1 / m)

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
        result1 = Solve(A_s, v, t0=t, D=D, lowerBound=0, upperBound=upper, plot=true, linesearch=linesearch, ε=ε, numSample=1, mode=mode, logfilename=outputfile, startεd0=startεd0)
    elseif mode == "C"
        result1 = Solve(C, v, t0=t, D=D, lowerBound=0, upperBound=upper, plot=true, linesearch=linesearch, ε=ε, numSample=1, mode=mode, logfilename=outputfile, startεd0=startεd0)
    end

    return (v=result1.v, t=result1.t, z=result1.z)
end

#=
opt = 12083.2
ε1 = 10^(-1.5)
inputfile = "C:/Users/pchib/Desktop/MASTER/MESDP/Gset/g1.txt"
#disp(@benchmark exp1(inputfile, nothing, opt, nothing, ε=ε1, linesearch=true, bound=true, color=:black, mode="A"))
u = @benchmark exp1(inputfile, nothing, opt, nothing, ε=ε1, linesearch=true, bound=true, color=:black, mode="C")
println(mean(u).memory)
println(mean(u).time)
=#
#=
ε1 = 10^(-2)
inputfile = "Gset/G49.txt"
=#
#=
outputfile = "MATLABplot/G48log-0.txt"
exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=0.0)


outputfile = "MATLABplot/G48log-1.txt"
exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=-1.0)


outputfile = "MATLABplot/G48log-2.txt"
exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=-2.0)
=#
#=
outputfile = "MATLABplot/G49log-3.txt"
exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=-3.0)
=#
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
global colorcnt = 0
cnt = 29
ε1 = 10^(-2.5);
for i in 1:cnt
    inputfile = "Gset/g" * string(num[i]) * ".txt"
    outputfile = "Result/exp7/G" * string(num[i]) * "log.txt"
    exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=0.0)
end
#=
for i in 1:cnt
    inputfile = "C:/Users/pchib/Desktop/MASTER/MESDP/Gset/g" * string(num[i]) * ".txt"
    #inputfile = "C:/Users/pchib/Desktop/MASTER/MESDP/Gset/g3k" * string(num[i]) * ".txt"
    opt = optval[i]
    #print(string(num[i]) * ": ")
    if i in labeli
        global colorcnt = colorcnt + 1
        idx = findall(x -> x == i, labeli)
        lbl = label[idx]
        u = @benchmark exp1(inputfile, nothing, opt,
            lbl[1], ε=ε[i] / sqrt(2), linesearch=true, bound=true, color=colors[colorcnt], mode="C")
        println(mean(u).memory, " \n", mean(u).time)
        println()
    else
        lbl = nothing
        u = @benchmark exp1(inputfile, nothing, opt,
            nothing, ε=ε[i] / sqrt(2), linesearch=true, bound=true, color=colors[colorcnt], mode="C")
        println(mean(u).memory, " \n", mean(u).time)
    end

end
=#
#

#=
inputfile = "C:/Users/pchib/Desktop/MASTER/MESDP/Gset/g22.txt"
opt = 14135.95
result1 = exp1(inputfile, outputfile, opt,
"g22", ε=ε1, linesearch=false, bound=true)

inputfile = "C:/Users/pchib/Desktop/MASTER/MESDP/Gset/g51.txt"
opt = 4006.26
result1 = exp1(inputfile, outputfile, opt,
"g51", ε=ε1, linesearch=false, bound=true)
=#
