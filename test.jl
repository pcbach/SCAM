include("MESDP.jl")
include("ReadGSet.jl")
using BenchmarkTools
A, C = readfile("Gset/g1.txt")
#disp(A)
#disp(C)
n = size(C, 1)
m = size(A, 2)
v = B(A, d=1 / m)

u = @benchmark a1, b1, c1 = ArnoldiGrad(C, v, lowerBound=1.2, upperBound=1.3, D=ones(n, 1), mode="C")
display(mean(u).time)
display(mean(u))