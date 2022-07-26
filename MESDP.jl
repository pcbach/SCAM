#include("./Arnoldi/ArnoldiMethod.jl-master/src/ArnoldiMethodMod.jl")
include("Arnoldi/ArnoldiMethodMod.jl")
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
#Julia Display with endline
function disp(quan; name="")
    #print("____________________________\n")
    if name != ""
        print(name, ":\n")
    end
    display(quan)
    #print("\n____________________________")
    print("\n")
end

#==========================BASIC=ALGORITHM======================#
#calculate gradient matrix n x 
function grad(A; P=nothing, v=nothing)
    r = spzeros(m, m)
    for i in 1:n
        if P !== nothing
            a = A[:, i]
            r = r - (a' * P' * a)^-0.5 / 2 * a * a'
        elseif v !== nothing
            a = A[:, i]
            r = r - (a' * P' * a)^-0.5 / 2 * a * a'
        else
            println("grad error: No valid input")
        end
    end
    return r
end

function B(A; P=nothing, v=nothing)
    rows = rowvals(A)
    vals = nonzeros(A)
    r = spzeros(n)

    #prioratise v
    if P !== nothing && v !== nothing
        P = nothing
    end

    for i in 1:n
        if P !== nothing
            a = A[:, i]
            r[i] = a' * P' * a
        elseif v !== nothing
            for k in nzrange(A, i)
                r[i] += (vals[k] * v[rows[k]])
            end
            r[i] = r[i]^2
        else
            println("B error: No valid input")
        end

    end
    return r
end

#linear map 's gradient 
function ∇g(v; lowerBound=0, upperBound=1e16, D=ones(1, n))
    res = zeros(n)
    for i in 1:n
        res[i] = clamp(1 / (2 * ((v[i])^(1 / 2))), lowerBound / D[i], upperBound / D[i])
    end
    return res
end

#Adjoint linear map
function Badj(A, w)
    r = spzeros(m, m)
    for i in 1:n
        a = A[:, i]
        r = r + w[i] * a * a'
    end
    return r
end

#f = ∑√v
function f(A, v)
    n = size(A, 2)
    r = 0
    for i in 1:n
        r = r - sqrt(v[i])
    end
    return abs(r)
end

#Calculate true gradient
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
    q = B(A, v=w)
    return w, q, eig[1]
end

#Frank Wolfe with true gradient
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

function ArnoldiGrad(A, v; lowerBound=0, upperBound=1e16, tol=1e-5, D=ones(1, n))
    decomp, history = partialschur(A, v, tol=tol, which=LM(), lowerBound=lowerBound, upperBound=upperBound, D=D)#, mindim=6, maxdim=12)
    eig, eigv = partialeigen(decomp)
    w = eigv[:, 1]
    w = w / norm(w)
    q = B(A, v=w)
    return w, q, eig[1]
end

function gammaLineSearch(A, v, q; ε=1e-8)
    b = 0
    e = 1
    while e - b > ε
        mid1 = b + (e - b) / 3
        mid2 = e - (e - b) / 3
        vmid1 = (1 - mid1) * v + mid1 * q
        vmid2 = (1 - mid2) * v + mid2 * q
        if f(A, vmid1) < f(A, vmid2)
            b = mid1
        else
            e = mid2
        end
    end
    return (e + b) / 2
end

function CutValue(A, z)
    rows = rowvals(A)
    vals = nonzeros(A)
    r = spzeros(n)

    for i in 1:n
        for k in nzrange(A, i)
            r[i] += (vals[k] * z[rows[k]])
        end
        r[i] = sign(r[i])
    end
    #disp(r)
    #disp(r' * A' * A * r / 2)
    return r' * A' * A * r

end

function Solve(A, v0; D=ones((1, n)), t0=2, ε=1e-3, lowerBound=0, upperBound=1e16, printIter=false, plot=false, linesearch=false, numSample=1)
    v = v0
    t = t0
    z = rand(Normal(0, sqrt(1 / m)), (numSample, m))
    start = t0
    if !linesearch
        gamma = 2 / (t + start)
    end
    if plot
        xlog = zeros(0)
        ylog = zeros(0)
        timelog = zeros(0)
        gammalog = zeros(0)
        mingrad = zeros(0)
        maxgrad = zeros(0)
    end
    w, q, λ = ArnoldiGrad(A, v, lowerBound=lowerBound, upperBound=upperBound, D=D, tol=ε)
    gap = dot(q - v, ∇g(v, lowerBound=lowerBound, upperBound=upperBound, D=D)) / abs(f(A, v))
    εd0 = 0
    if plot
        append!(ylog, abs(f(A, v)))
    end

    while gap > ε
        if plot
            append!(ylog, abs(f(A, v)))
            #append!(timelog, time())
            #append!(gammalog, gammaLineSearch(A, v, q))
            #grad = ∇g(v, lowerBound=lowerBound, upperBound=upperBound, D=D)
            #append!(mingrad, minimum(grad))
            #append!(maxgrad, maximum(grad))
        end
        t = t + 1

        #disp(gammaLineSearch(v, q))
        if linesearch
            gamma = gammaLineSearch(A, v, q)
        end
        v = (1 - gamma) * v + gamma * q

        for i in 1:numSample
            z[i, :] = sqrt(1 - gamma) * z[i, :] + sqrt(gamma) * w * rand(Normal(0, 1))
        end

        if !linesearch
            gamma = 2 / (t + start)
        end
        w, q, λ = ArnoldiGrad(A, v, lowerBound=lowerBound, upperBound=upperBound, D=D, tol=ε)
        gap = dot(q - v, ∇g(v, lowerBound=lowerBound, upperBound=upperBound, D=D)) / abs(f(A, v))
        if gap < 10^(εd0)
            bestRes = 0
            bestIdx = 0
            for i in 1:numSample
                cut = CutValue(A, z[i, :])
                if cut > bestRes
                    bestRes = cut
                    bestIdx = i
                end
            end
            cutValue = CutValue(A, z[bestIdx, :]) / 2
            println(t, ": ", round.(abs(f(A, v)); digits=3), " ", round.(log10(abs(gap)); digits=3), " ", cutValue)
            εd0 = εd0 - 0.1
        elseif (t % 10 == 0)
            print('-')
        end
    end
    if plot
        bestRes = 0
        bestIdx = 0
        for i in 1:numSample
            cut = CutValue(A, z[i, :])
            if cut > bestRes
                bestRes = cut
                bestIdx = i
            end
        end
        if plot
            append!(ylog, abs(f(A, v)))
            #append!(timelog, time())
            #append!(gammalog, gammaLineSearch(A, v, q))
            #grad = ∇g(v, lowerBound=lowerBound, upperBound=upperBound, D=D)
            #append!(mingrad, minimum(grad))
            #append!(maxgrad, maximum(grad))
        end
        cutValue = Int64(CutValue(A, z[bestIdx, :]) / 2)
        result = (val=f(A, v), v=v, t=t, plot=(x=xlog, y=ylog), time=timelog, gamma=gammalog, mingrad=mingrad, maxgrad=maxgrad, z=z[bestIdx, :])
        return result
    else
        bestRes = 0
        bestIdx = 0
        for i in 1:numSample
            cut = CutValue(A, z[i, :])
            if cut > bestRes
                bestRes = cut
                bestIdx = i
            end
        end
        result = (val=f(A, v), v=v, t=t, z=z[bestIdx, :])
        return result
    end

end