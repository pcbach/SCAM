using LinearAlgebra
using Distributions
using Plots
using BenchmarkTools
using Statistics
using Arpack
using DelimitedFiles, CSV, Tables
using DataStructures
using Measures
#using PlotlyJS

#n = 0 #vertex
#m = 0 #edge
Result = 0 #edge
lanc2cnt = 0
lanccnt = 0
lanc1 = zeros(0)
lanc2 = zeros(0)

#==============================MISC=============================#
function disp(quan; name="")
    if name != ""
        print(name, ":\n")
    end
    display(quan)
    print('\n')
end
#==============================GRAPH============================#
function initGraphAdj(edgeList)
    Adj = Any[]
    for i in 1:n
        push!(Adj, [])
    end
    for i in 1:m
        v1 = minimum(edgeList[i, :])
        v2 = maximum(edgeList[i, :])
        push!(Adj[v1], i)
        push!(Adj[v2], -i)
    end
    return Adj
end

function initGraphA(edgeList)
    A = zeros(m, n)
    for i in 1:m
        u = edgeList[i, 1]
        v = edgeList[i, 2]
        A[i, u] = 1
        A[i, v] = -1
    end
    return A
end
#==========================BASIC=ALGORITHM======================#
function grad_(A, P)
    r = zeros(m, m)
    for i in 1:n
        a = A[:, i]
        r = r - (a' * P' * a)^-0.5 / 2 * a * a'
    end
    return r
end

function grad(A, v)
    r = zeros(m, m)
    for i in 1:n
        a = A[:, i]
        r = r - a * a' / (2 * sqrt(v[i]))
    end
    return r
end

function B(A, w)
    r = zeros(n)
    for i in 1:n
        a = A[:, i]
        r[i] = (a'*w)[1, 1]^2
    end
    return r
end

function B_(A, P)
    r = zeros(n)
    for i in 1:n
        a = A[:, i]
        r[i] = a' * P' * a
    end
    return r
end

function ∇g(v)
    return 1 ./ (4 .* ((v) .^ (3 / 4)))
    #return 1 ./ (2 .* ( (v).^(1/2)))
end

function Badj(A, w)
    r = zeros(m, m)
    for i in 1:n
        a = A[:, i]
        r = r + w[i] * a * a'
    end
    return r
end

function f(A, v)
    n = size(A, 2)
    r = 0
    for i in 1:n
        r = r - sqrt(v[i])
        #r = r - v[i].^(1/4)
    end
    return abs(r)
end

function LMOtrueGrad(A, v)
    ∇P = zeros(m, m)
    for i in 1:n
        a = A[:, i]
        ∇P = ∇P - a * a' / (2 * sqrt(v[i]))
        #∇P = ∇P - a*a'/(4*v[i]^(3/4))
    end
    (eig, eigv) = eigen(∇P)
    w = eigv[:, 1]
    w = w / norm(w)
    q = B(A, w)
    return w, q, eig[1]
end

function solvesampTrue(A, v0, z0)
    v = v0
    z = z0
    t = 0
    start = 10
    gamma = 2 / (t + start)
    w, q, λ = LMOtrueGrad(A, v)
    flag = true
    while dot(q - v, ∇g(v)) / abs(f(A, v)) > 1e-4
        value = abs(f(A, v))
        print(t, "-", dot(q - v, ∇g(v)) / value, " ", value, "\n")
        append!(lanc2, value)
        #=if t > 30 && flag && (lanc2[t+1] - lanc2[t-10]) < 0.5
            plot([t+1],[value],shape = :x)
            flag = false
        end=#
        t = t + 1
        z = sqrt(1 - gamma) * z + sqrt(gamma) * w * rand(Normal(0, 1))
        v = (1 - gamma) * v + gamma * q

        gamma = 2 / (t + start)
        w, q, λ = LMOtrueGrad(A, v)
    end
    x = sign.(A' * z)
    result = (val=f(A, v), x=x, v=v, z=z)
    return result
end

#===========================GEN=DATA============================#
function genSample(A)
    p = ones((n, 1))
    p[1] = -1
    p = A * p
    p = p / norm(p)
    X0 = p * p'
    val, vec = eigen(X0)
    val = max.(1e-7, val)
    X0 = Matrix(Hermitian(vec * Diagonal(val) * inv(vec)))
    z = rand(MvNormal(zeros(m), X0))
    v = B_(A, X0)
    return z, v
end

function genSample2(A)
    p = rand(Uniform(-1, 1), (m, 1))
    p = p
    X0 = p * p'
    X0 = X0 / tr(X0)
    val, vec = eigen(X0)
    val = max.(1e-7, val)
    X0 = Matrix(Hermitian(vec * Diagonal(val) * inv(vec)))
    z = rand(MvNormal(zeros(m), X0))
    v = B_(A, X0)
    return z, v
end

function genSampleP(A, P)
    X0 = P / tr(P)
    val, vec = eigen(X0)
    val = max.(1e-10, val)
    X0 = Matrix(Hermitian(vec * Diagonal(val) * inv(vec)))
    print(size(X0), m)
    z = rand(MvNormal(zeros(m), X0))
    v = B_(A, X0)
    return z, v
end

#==============================MAIN=============================#
A = readdlm("graphbackup.csv", ',', Float64, '\n')
A = A / 2
#P = readdlm("P.csv", ',', Float64, '\n')
#global Result = readdlm("result.csv", ',', Float64, '\n')[1]
global m = size(A, 1)
global n = size(A, 2)

z, v = genSample(A)
result = solvesampTrue(A, v, z)
disp(result.v)
disp(sqrt.(result.v))
disp(result.x)
#=
x = [-0.5,3]
for i = -3:0.1:6
    y = x.*-2 .+ i
    plot!(x,y,lw = 0.2,color = :black,label = "",xlim = [0,2.5], ylim = [-2.5,0])
end
for i = -3:1:6
    y = x.*-2 .+ i
    plot!(x,y,lw = 2,color = :black,label = "",xlim = [0,2.5], ylim = [-2.5,0])
end
for i = 0:0.5:0
y = x.*-2 .+ i
plot!(x,y,lw = 2,color = :black,label = "",xlim = [0,2.5], ylim = [-2.5,0])
end


global lanc2 = zeros(0)
z,v = genSample(A)
solvesampTrue(A,v,z)
opt = lanc2[length(lanc2)]
plot!(log10.(1:1:length(lanc2)),log10.(abs.(opt.-lanc2)/opt),lw = 2,size = (2000,2000),dpi = 100,tickfontsize = 30,legend = false,gridlinealpha = 0.5,gridlinewidth = 2,color = :red)

for i in 1:5
    global lanc2 = zeros(0)
    z,v = genSample2(A)
    solvesampTrue(A,v,z)
    opt = lanc2[length(lanc2)]
    plot!(log10.(1:1:length(lanc2)),log10.(abs.(opt.-lanc2)/opt),lw = 2,size = (2000,2000),dpi = 100,tickfontsize = 30,legend = false,gridlinealpha = 0.5,gridlinewidth = 2,color = :blue)
end


#disp(lanc2)
#CSV.write("log10.csv",Tables.table(reshape(lanc2,(1,:))),append = true)
savefig("plot/plot1.png")
=#