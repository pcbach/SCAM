include("./Arnoldi/ArnoldiMethod.jl-master/src/ArnoldiMethodMod.jl")
using .ArnoldiMethodMod
using LinearAlgebra
using Distributions
#using Plots
#using BenchmarkTools
using Statistics
#using Arpack
#using DelimitedFiles, CSV, Tables
#using DataStructures
#using Measures
using SparseArrays

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
    A = spzeros(m, n)
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
    r = spzeros(m, m)
    for i in 1:n
        a = A[:, i]
        r = r - (a' * P' * a)^-0.5 / 2 * a * a'
    end
    return r
end

function grad(A, v)
    r = spzeros(m, m)
    for i in 1:n
        a = A[:, i]
        r = r - a * a' / (2 * sqrt(v[i]))
    end
    return r
end

function B(A, w)
    r = spzeros(n)
    for i in 1:n
        a = A[:, i]
        r[i] = real((a'*w)[1, 1]^2)
    end
    return r
end

function B_(A, P)
    r = spzeros(n)
    for i in 1:n
        a = A[:, i]
        r[i] = a' * P' * a
    end
    return r
end

function cap(val, a)
    if val > 2 * a
        #println("capped up")
        val = 2 * a
    elseif val < 1 / (2 * a)
        val = 1 / (2 * a)
        #println("capped low")
    end
    return val
end

function ∇g(v; lowerBound=0, upperBound=1e16)
    grad_ = 1 ./ (2 .* ((v) .^ (1 / 2)))
    res = clamp.(grad_, lowerBound, upperBound)
    #println(sum(res .== lowerBound), '-', sum(res .== upperBound))
    return res
end

function Badj(A, w)
    r = spzeros(m, m)
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


function LMOtrueGrad(A, v; lowerBound=0, upperBound=1e16)
    ∇P = spzeros(m, m)
    for i in 1:n
        a = A[:, i]
        grad_ = 1 / (2 * sqrt(v[i]))
        grad_ = clamp(grad_, lowerBound, upperBound)
        ∇P = ∇P - a * a' * grad_
    end
    (eig, eigv) = eigen(∇P)
    w = eigv[:, 1]
    w = w / norm(w)
    q = B(A, w)
    return w, q, eig[1]
end


function solvesampTrue(A, v0; t0=2, ε=1e-3, lowerBound=0, upperBound=1e16)
    v = v0
    t = 0
    start = t0
    gamma = 2 / (t + start)
    w, q, λ = LMOtrueGrad(A, v, lowerBound=lowerBound, upperBound=upperBound)
    while dot(q - v, ∇g(v, lowerBound=lowerBound, upperBound=upperBound)) / abs(f(A, v)) > ε
        t = t + 1
        v = (1 - gamma) * v + gamma * q
        gamma = 2 / (t + start)
        w, q, λ = LMOtrueGrad(A, v, lowerBound=lowerBound, upperBound=upperBound)

    end
    result = (val=f(A, v), t=t)
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

function ArnoldiGrad(A, v; lowerBound=0, upperBound=1e16, tol=1e-3)
    decomp, history = partialschur(A, v, tol=tol, which=LM(), lowerBound=lowerBound, upperBound=upperBound, mindim=6, maxdim=12)
    eig, eigv = partialeigen(decomp)
    w = eigv[:, 1]
    w = w / norm(w)
    q = B(A, w)
    return w, q, eig[1]
end

function Solve(A, v0; t0=2, ε=1e-3, lowerBound=0, upperBound=1e16, printIter=false, plot=false)
    v = v0
    t = 0
    start = t0
    gamma = 2 / (t + start)
    if plot
        tally = zeros(0)
        timelog = zeros(0)
    end
    w, q, λ = ArnoldiGrad(A, v, lowerBound=lowerBound, upperBound=upperBound, tol=ε)
    gap = dot(q - v, ∇g(v, lowerBound=lowerBound, upperBound=upperBound)) / abs(f(A, v))
    while gap > ε
        if printIter
            println(t, ": ", abs(f(A, v)), " ", gap)
        end
        if plot
            append!(tally, abs(f(A, v)))
            append!(timelog, time())
        end
        t = t + 1
        v = (1 - gamma) * v + gamma * q

        gamma = 2 / (t + start)
        w, q, λ = ArnoldiGrad(A, v, lowerBound=lowerBound, upperBound=upperBound, tol=ε)
        gap = dot(q - v, ∇g(v, lowerBound=lowerBound, upperBound=upperBound)) / abs(f(A, v))
    end
    if plot
        result = (val=f(A, v), t=t, plot=tally, time=timelog)
        return result
    else
        result = (val=f(A, v), t=t)
        return result
    end

end