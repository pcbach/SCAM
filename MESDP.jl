
include("Arnoldi/ArnoldiMethodMod.jl")
using .ArnoldiMethodMod
using LinearAlgebra
using Distributions
using Statistics
using SparseArrays

#==============================MISC=============================#
#Julia Display with endline
function disp(quan; name="")
    if name != ""
        print(name, ":\n")
    end
    display(quan)
    print('\n')
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

#Linear map
function B(A; P=nothing, v=nothing, d=nothing)
    rows = rowvals(A)
    vals = nonzeros(A)
    r = spzeros(n)

    #prioratise v
    if P !== nothing && v !== nothing
        P = nothing
    end

    for i in 1:n
        if d !== nothing
            for k in nzrange(A, i)
                r[i] += vals[k]^2
            end
            r[i] = r[i] * d
        elseif P !== nothing
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
function f(v)
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
    while dot(q - v, ∇g(v, lowerBound=lowerBound, upperBound=upperBound)) / abs(f(v)) > ε
        t = t + 1
        v = (1 - gamma) * v + gamma * q
        gamma = 2 / (t + start)
        w, q, λ = LMOtrueGrad(A, v, lowerBound=lowerBound, upperBound=upperBound)

    end
    result = (val=f(A), t=t)
    return result
end

#LMO
function ArnoldiGrad(A, v; lowerBound=0, upperBound=1e16, tol=1e-2, D=ones(1, n), mode="A")
    Mn = lowerBound ./ D
    Mx = upperBound ./ D
    λ = sqrt.(clamp.(1 ./ (2 .* sqrt.(v)), Mn, Mx))
    if mode == "A"
        decomp, history = partialschur(A, λ, tol=tol, which=LM(), mode=mode)
        eig, eigv = partialeigen(decomp)
        w = eigv[:, 1]
        q = B(A, v=w)
        return w, q, eig[1]
    elseif mode == "C"
        decomp, history = partialschur(A, λ, tol=tol, which=LM(), mode=mode)
        eig, eigv = partialeigen(decomp)
        u = real(eigv[:, 1])
        tmp = zeros(n, 1)
        tmp2 = zeros(n, 1)
        tmp = λ .* u
        tmp2 = A * tmp
        tmp = λ .* tmp2
        scale = (tmp'*u)[1]
        tmp2 = tmp2 .^ 2
        tmp2 /= scale
        return u, tmp2, eig[1]
    end
end

#linesearch 
function gammaLineSearch(v, q; ε=1e-8)
    b = 0
    e = 1
    while e - b > ε
        #println(b, " ", e)
        # Find the mid1 and mid2
        mid1 = b + (e - b) / 3
        mid2 = e - (e - b) / 3
        vmid1 = (1 - mid1) * v + mid1 * q
        vmid2 = (1 - mid2) * v + mid2 * q
        #disp(vmid1)
        #disp(vmid2)
        if f(vmid1) < f(vmid2)
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
    return r' * A' * A * r / 2

end

function Solve(A, v0; D=ones((1, n)), t0=2, ε=1e-3, lowerBound=0, upperBound=1e16, plot=false, linesearch=false, numSample=1, mode="A", logfilename=nothing, startεd0=-3.0)
    if logfilename !== nothing
        open(logfilename, "w") do io
            print()
        end
    end
    v = v0
    t = t0
    z = rand(Normal(0, 1 / m), (numSample, m))
    start = 0
    if !linesearch
        gamma = 2 / (t + start)
    end
    if plot
        flog = zeros(0)
        glog = zeros(0)
    end
    εd0 = startεd0
    w, q, λ = ArnoldiGrad(A, v, lowerBound=lowerBound, upperBound=upperBound, D=D, mode=mode, tol=10^(εd0))
    gap = dot(q - v, ∇g(v, lowerBound=lowerBound, upperBound=upperBound, D=D)) / abs(f(v))
    while gap > ε
        if plot
            append!(flog, abs(f(v)))
            append!(glog, gap)
        end

        if linesearch && t > 10
            gamma = gammaLineSearch(v, q)
        else
            gamma = 2 / (t + start)
        end
        t = t + 1
        v = (1 - gamma) * v + gamma * q

        if !linesearch
            gamma = 2 / (t + start)
        end
        w, q, λ = ArnoldiGrad(A, v, lowerBound=lowerBound, upperBound=upperBound, D=D, mode=mode, tol=10^(εd0))
        gap = dot(q - v, ∇g(v, lowerBound=lowerBound, upperBound=upperBound, D=D)) / abs(f(v))
        #println(t, " ", abs(f(v)), " ", gap)
        if logfilename !== nothing
            open(logfilename, "a") do io
                println(io, gap)
            end
        end

        if gap < 10^(εd0)
            εd0 -= 1
            println("Change accuracy to ", 10^(εd0))
        end

    end
    if plot
        append!(flog, abs(f(v)))
        append!(glog, gap)
        bestRes = 0
        bestIdx = 0
        for i in 1:numSample
            cut = CutValue(A, z[i, :])
            if cut > bestRes
                bestRes = cut
                bestIdx = i
            end
        end
        result = (val=f(v), v=v, t=t, plot=flog, z=z[bestIdx, :], gap=glog)
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
        result = (val=f(v), v=v, t=t, z=z[bestIdx, :])
        return result
    end

end