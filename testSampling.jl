include("MESDP.jl")
include("ReadGSet.jl")
using LaTeXStrings
using Plots
using Plots.PlotMeasures
using BenchmarkTools

#Single graph with individual bound on the gradient
function exp1(inputFile, outputfile; linesearch=false, ε=1e-2, v0=nothing, t0=0, bound=true, mode="A", startεd0=0.0)
    file = inputFile
    print("Readfile ")
    @time A, C = readfile(file)
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

#=========================================================================================
=#
ε1 = 10^(-2)

file = ["1e4n3d.txt", "1e4n3d2.txt", "1e4n5d.txt", "1e4n5d2.txt", "1e4n10d.txt", "1e4n10d2.txt"]

for i = 1:length(file)
    inputfile = "BigExample/" * file[i]
    outputfile = "Result/exp5/" * chop(file[i], tail=4) * "log.txt"
    println(inputfile)
    println(outputfile)
    print("Solve ")
    @time exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=0.0)
end

#=========================================================================================

outputfile = "MATLABplot/G48log-1.txt"
exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=-1.0)


outputfile = "MATLABplot/G48log-2.txt"
exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=-2.0)
=#
#=
outputfile = "MATLABplot/G49log-3.txt"
exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=-3.0)
=#

