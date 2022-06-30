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
    rows = nothing
    vals = nothing

    p = rand(Uniform(-1, 1), (n, 1))

    p = A_s * p
    p = p / norm(p)
    v = B(A_s, v=p)
    p = nothing
    t = 2
    opt = sqrt(optimalValue * 2)
    if v0 !== nothing
        v = v0
        t = t0
    end
    result1 = Solve(A_s, v, t0=t, D=D, lowerBound=opt, upperBound=opt * 2, printIter=true, plot=true, linesearch=linesearch, ε=ε)
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
    disp(result1.val^2 / 2)
    if linesearch
        color = :red
    else
        color = :blue
    end

    if t0 == 0
        style = :solid
    else
        style = :dash
    end
    plot!(log10.(t0 - 1 .+ (1:length(result1.plot))), log10.((opt .- result1.plot) ./ opt), ratio=:equal, label=label, dpi=300, size=(1000, 1000),
        lw=3, lc=color, ls=style, legend_font_pointsize=20, tickfontsize=20, legend_position=:bottomleft)
    if outputFile !== nothing
        savefig(outputFile)
    end
    return (v=result1.v, t=result1.t)
end
#"C:/Users/pchib/Desktop/MASTER/MESDP/toy.txt"
#print(m,n)
opt = 4006.2
ε1 = 1e-2
ε2 = 1e-3
inputfile = "C:/Users/pchib/Desktop/MASTER/MESDP/Gset/g54.txt"
outputfile = "C:/Users/pchib/Desktop/MASTER/MESDP/Result/exp1/G54.png"
result = exp1(inputfile, nothing, opt,
    "Linesearch Start", ε=ε1, linesearch=true)

exp1(inputfile, nothing, opt,
    "Linesearch", ε=ε2, v0=result.v, t0=result.t - 1, linesearch=true)

exp1(inputfile, nothing, opt,
    "Linesearch -> Schedule", ε=ε2, v0=result.v, t0=result.t - 1, linesearch=false)

result = nothing
result = exp1(inputfile, nothing, opt,
    "Schedule Start", ε=ε1, linesearch=false)

exp1(inputfile, nothing, opt,
    "Sechedule -> Linesearch", ε=ε2, v0=result.v, t0=result.t - 1, linesearch=true)

exp1(inputfile, outputfile, opt,
    "Schedule", ε=ε2, v0=result.v, t0=result.t - 1, linesearch=false)
#"C:/Users/pchib/Desktop/MASTER/MESDP/Gset/g5.txt"
#exp1("Gset/G14.txt", "Result/exp1/G14and5.png", 3191.6, "max")