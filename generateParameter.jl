using DelimitedFiles
m = readdlm("Count.csv", ',', Float64, '\n')[1]
writedlm("Count.csv", m + 1, ',')