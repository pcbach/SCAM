using LinearAlgebra
using Distributions
using Plots
using BenchmarkTools
using Statistics
using Arpack
using DelimitedFiles, CSV, Tables
using DataStructures
using CSV, Tables
File = CSV.File("log8.csv", header=false)
cnt = 0
para = readdlm("parameter.csv", ',', Float64, '\n')

x = [-0.5, 3]
for i = -3:0.1:6
    y = x .* -2 .+ i
    plot!(x, y, lw=0.2, color=:black, label="", xlim=[0, 2.5], ylim=[-2.5, 0])
end
for i = -3:1:6
    y = x .* -2 .+ i
    plot!(x, y, lw=2, color=:black, label="", xlim=[0, 2.5], ylim=[-2.5, 0])
end
for i = 0:0.5:0
    y = x .* -2 .+ i
    plot!(x, y, lw=2, color=:black, label="", xlim=[0, 2.5], ylim=[-2.5, 0])
end

scatter!(
    x=[0, 0.5, 1, 1.5, 2, 2.5],
    y=[0, 0, 0, 0, 0, 0],
    mode="text",
    name="",
    text=["0", "1", "2", "3", "4", "5"],
    textposition="top center"
)
for row in File
    plotdata = zeros(0)
    for cell in row
        if typeof(cell) == Float64
            append!(plotdata, Float64(cell))
        end
    end
    i = Int(floor(cnt / 10))
    print(i, "- ", cnt, " \n")
    color = :black
    style = :dash
    if cnt == 89
        i = 9
    end
    if i < 3
        color = :red
    elseif i < 6
        color = :blue
    elseif i < 9
        color = :green
    else
        color = :orange
    end


    if i % 3 == 0
        style = :dash
    elseif i % 3 == 1
        style = :dot
    else
        style = :solid
    end
    xs = 1:length(plotdata)
    n = Int(para[i+1, 1])
    m = Int(para[i+1, 2])
    if (cnt % 10 == 0)
        f1 = plot!(log10.(xs), plotdata, lw=2, label="n = " * string(n) * ", m= " * string(m), aspect_ratio=:equal, size=(2000, 2000), dpi=100, tickfontsize=30, gridlinewidth=2, c=color, ls=style, legendfontsize=20, legend=:bottomleft)
    else
        f1 = plot!(log10.(xs), plotdata, lw=2, label="", aspect_ratio=:equal, size=(2000, 2000), dpi=100, tickfontsize=30, gridlinewidth=2, c=color, ls=style)
    end
    global cnt += 1
end


savefig("plot/plotconvergence.png")