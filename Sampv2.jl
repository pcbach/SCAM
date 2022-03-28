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
    return 1 ./ (2 .* (.√(v)))
    #return 1 ./ (3 .* ( (v).^(2/3) ))
end

function Badj(A, w)
    r = zeros(m, m)
    for i in 1:n
        a = A[:, i]
        r = r + w[i] * a * a'
    end
    return r
end

function f_(A, P)
    n = size(A, 2)
    r = 0
    for i in 1:n
        a = A[:, i]
        r = r - sqrt(a' * P * a)
    end
    return r
end

function f(A, v)
    n = size(A, 2)
    r = 0
    for i in 1:n
        r = r - sqrt(v[i])
        # r = r - v[i].^(1/3)
    end
    return r
end
#============================FRANK=WOLFE=========================#
function LMOFW(A, P)
    ∇P = zeros(m, m)
    for i in 1:n
        a = A[:, i]
        ∇P = ∇P - (a' * P * a)^-0.5 / 2 * a * a'
    end
    (eig, eigv) = eigen(∇P)
    w = eigv[:, 1]
    w = w / norm(w)
    q = B(A, w)
    return w
end

function LMOtrueGrad(A, v)
    ∇P = zeros(m, m)
    for i in 1:n
        a = A[:, i]
        ∇P = ∇P - a * a' / (2 * sqrt(v[i]))
        #∇P = ∇P - a*a'/(3*v[i]^(2/3))
    end
    (eig, eigv) = eigen(∇P)
    w = eigv[:, 1]
    w = w / norm(w)
    q = B(A, w)
    return w, q, eig[1]
end

function solveFW(A, X0; debug=false)
    X = X0 / tr(X0)
    epoch = 0
    while true
        epoch = epoch + 1
        w = LMOFW(A, X)
        H = w * w'
        gamma = 2 / (epoch + 2)
        G = grad_(A, X)
        if debug
            print(epoch, ": ", f_(A, X)^2, ' ', dot(X - H, G), '\n')
        end
        if abs(dot(X - H, G)) < 1e-3
            break
        end
        X = (1 - gamma) * X + gamma * H
    end
    U, S, V = svd(X)
    y = U[:, 1] * sqrt(S[1])
    x = sign.(A' * y)
    return f_(A, X)^2, x
end
#============================POWER=ITER==========================#
function LMOPI(A, v, ϵ::Float64=1e-3)
    b = rand(Uniform(-1, 1), (m, 1))
    b = b / norm(b)
    w = zeros(m)
    epoch = 0
    while true
        w = zeros(m)
        for i in 1:n
            a = A[:, i]
            c1 = -1 / (2 * sqrt(v[i]))
            w_ = a' * b
            w_ = c1 * a * w_
            w = w + w_
        end
        w = w / norm(w)
        if ((1 - abs(dot(b, w))) < ϵ)
            break
        end
        b = copy(w)
        epoch = epoch + 1
    end
    w = w / norm(w)
    q = B(A, w)
    return w, q
end

function solvesampPow(A, v0, z0; debugw=false, plotcon=false, ϵ::Float64=1e-3, terminate=false, ϵ2::Float64=1e-3)
    v = v0
    z = z0
    t = 0
    gamma = 2 / (t + 2)
    w, q = LMOPI(A, v)
    if plotcon
        tally = zeros(0)
        tally2 = zeros(0)
        tallyv = Array(sort(v))
    end
    #disp(tallyv)
    if debugw
        disp(w, name="w")
    end
    #print((t,mvavg,buffersize,ϵ))
    while dot(q - v, ∇g(v)) > ϵ #=(terminate && t < 10000) ||=#
        error = dot(q - v, ∇g(v))
        if plotcon
            append!(tally, error)
            append!(tally2, (Result - f(A, v)^2) / Result)
        end
        t = t + 1
        z = sqrt(1 - gamma) * z + sqrt(gamma) * w * rand(Normal(0, 1))
        v = (1 - gamma) * v + gamma * q
        #tallyv = hcat(tallyv,sort(v))

        gamma = 2 / (t + 2)
        w, q = LMOPI(A, v, ϵ2)
        if debugw
            disp(w, name="w")
        end
    end

    #disp(tallyv)
    x = sign.(A' * z)
    result = (val=f(A, v)^2, x=x, v=v, z=z)
    append!(pow, f(A, v)^2)
    if plotcon
        plot!(log10.(abs.(tally)))
        #plot!(log10.(tally2))
        #disp(log10.(tally2[2500]))
        #=for i in 1:n
            plot!(log10.(tallyv[i,:]))
        end=#
    end
    return result
end

#==============================LANCZOS===========================#
function lanczosSubRoutine(A, v, b)
    wp = zeros(m)
    for i in 1:n
        a = A[:, i]
        c = -1 / (2 * sqrt(v[i]))
        w_ = a' * b
        w_ = c * a * w_
        wp = wp + w_
    end
    return wp
end

function lanczos(A, v)
    m = size(A, 1)
    k = m
    Q = zeros(m, k)
    Al = zeros(k)
    β = zeros(k)
    W = zeros(m, k)
    q = rand(Uniform(-1, 1), (m, 1))
    q = q / norm(q)
    Q[:, 1] = q
    W[:, 1] = lanczosSubRoutine(A, v, Q[:, 1])
    Al[1] = Q[:, 1]' * W[:, 1]
    W[:, 1] = W[:, 1] - Al[1] * Q[:, 1]
    T = zeros(k, k)
    T[1, 1] = Al[1]
    for i in 2:k
        β[i] = norm(W[:, i-1])
        if β[i] == 0
            break
        else
            Q[:, i] = W[:, i-1] / β[i]
        end
        W[:, i] = lanczosSubRoutine(A, v, Q[:, i])
        Al[i] = Q[:, i]' * W[:, i]
        W[:, i] = W[:, i] - Al[i] * Q[:, i] - β[i] * Q[:, i-1]
    end
    return Al, β
end

function lanczosv2(A, v; w=rand(Uniform(-0.1, 0.1), (m, 1)), ϵ::Float64=1e-6, ψ::Float64=3e-2, freq::Int64=10)
    m = size(A, 1)
    n = size(A, 2)
    Q = zeros(m, m)
    Al = zeros(m)
    β = zeros(m)
    QCurr = 0
    QPrev = 0
    W = w
    maxepoch = 1
    T = zeros(1, 1)
    for j in 1:m
        if j == 1
            QCurr = W / norm(W)
        else
            QCurr = W / β[j-1]
        end

        W = lanczosSubRoutine(A, v, QCurr)

        if j > 1
            W = W - QPrev * β[j-1]
        end

        Al[j] = (QCurr'*W)[1, 1]

        W = W - QCurr * Al[j]

        β[j] = norm(W)

        QPrev = copy(QCurr)
        Q[:, j] = QCurr
    end
    return Al, β, Q
end

function lanczosExperimental(A, v, λ; w=rand(Uniform(-0.1, 0.1), (m, 1)), ϵ::Float64=1e-6, ψ::Float64=3e-2, freq::Int64=10)
    m = size(A, 1)
    α = zeros(m)
    β = zeros(m)
    Tvector = zeros(m)
    vector = zeros(m)
    QOdd = 0
    QEven = 0
    W = w#rand(Uniform(-0.1, -0.1),(m,1))
    T = zeros(1, 1)
    Tvector[1] = 1
    #w[2] = -(α[1])/β[1]
    for j in 1:10
        #Normal Lanczos Routine
        if j % 2 == 0
            QOdd = W / β[j-1]
            W = lanczosSubRoutine(A, v, QOdd)
            if j > 1
                W = W - QEven * β[j-1]
            end
            α[j] = (QOdd'*W)[1, 1]
            W = W - QOdd * α[j]
            β[j] = norm(W)
        else
            if j == 1
                QEven = W / norm(W)
            else
                QEven = W / β[j-1]
            end
            W = lanczosSubRoutine(A, v, QEven)
            if j > 1
                W = W - QOdd * β[j-1]
            end
            α[j] = (QEven'*W)[1, 1]
            W = W - QEven * α[j]
            β[j] = norm(W)
        end

        if j == 2
            Tvector[2] = -(α[1] - λ) / β[1]
        elseif j == m
            Tvector[m] = -β[m-1] * Tvector[m-1] / (α[m] - λ)
        elseif j > 1
            Tvector[j+1] = -(β[j-1] * vector[j-1] + (α[j] - λ) * vector[j]) / β[j]
        end

        for i in 1:m
            if j % 2 == 0
                vector += Tvector[j] * QOdd
            else
                vector += Tvector[j] * QEven
            end
        end
    end
    vector ./= norm(Tvector)
    return vector
end

function solvesampLanc2(A, v0, z0; debugw=false, plotcon=false, ϵ::Float64=1e-3, ϵ2::Float64=1e-3)
    m = size(A, 1)
    n = size(A, 2)
    v = v0
    z = z0
    t = 0
    gamma = 2 / (t + 2)
    w, q = lanczosv2(A, v)
    if plotcon
        tally = zeros(0)
        tally2 = zeros(0)
        tallyv = Array(sort(v))
    end
    #disp(tallyv)
    if debugw
        disp(w, name="w")
    end
    #print((t,mvavg,buffersize,ϵ))
    while dot(q - v, ∇g(v)) > ϵ
        #print(t,'-')
        error = dot(q - v, ∇g(v))
        if plotcon
            append!(tally, error)
            #print(round(abs(Result - f(A,v)^2)/Result),"-")
            append!(lanc2, log(abs(Result - f(A, v)^2) / Result))
        end
        t = t + 1
        z = sqrt(1 - gamma) * z + sqrt(gamma) * w * rand(Normal(0, 1))
        v = (1 - gamma) * v + gamma * q
        #tallyv = hcat(tallyv,sort(v))

        gamma = 2 / (t + 2)
        w, q = lanczosv2(A, v)
        #disp(w)
        #println("##########################")
        if debugw
            disp(w, name="w")
        end
    end

    #disp(tallyv)
    x = sign.(A' * z)
    result = (val=f(A, v)^2, x=x, v=v, z=z)
    #append!(lanc2,f(A,v)^2)
    if plotcon
        #plot!(log10.(abs.(tally)))
        plot!(log10.(tally2))
        #disp(log10.(tally2[2500]))
        #=for i in 1:n
            plot!(log10.(tallyv[i,:]))
        end=#
    end
    return result
end

function solvesampLanc(A, v0, z0; debugw=false, plotcon=false, ϵ::Float64=1e-3, ϵ2::Float64=1e-3)
    m = size(A, 1)
    n = size(A, 2)
    v = v0
    z = z0
    t = 0
    gamma = 2 / (t + 2)
    w, q = lanczos(A, v)
    if plotcon
        tally = zeros(0)
        tally2 = zeros(0)
        tallyv = Array(sort(v))
    end
    #disp(tallyv)
    if debugw
        disp(w, name="w")
    end
    #print((t,mvavg,buffersize,ϵ))
    while dot(q - v, ∇g(v)) > ϵ
        #print(t,'-')
        error = dot(q - v, ∇g(v))
        if plotcon
            #append!(tally,error)
            #append!(tally2,abs(Result - f(A,v)^2)/Result)
            #append!(lanc2,log(abs(Result-f(A,v)^2)/Result) )
            append!(lanc2, log(abs(sqrt(Result) - f(A, v)) / sqrt(Result)))
        end
        t = t + 1
        z = sqrt(1 - gamma) * z + sqrt(gamma) * w * rand(Normal(0, 1))
        v = (1 - gamma) * v + gamma * q
        #tallyv = hcat(tallyv,sort(v))

        gamma = 2 / (t + 2)
        w, q = lanczos(A, v)
        if debugw
            disp(w, name="w")
        end
    end

    #disp(tallyv)
    x = sign.(A' * z)
    result = (val=f(A, v)^2, x=x, v=v, z=z)
    #append!(lanc1,f(A,v)^2)
    if plotcon
        #plot!(log10.(abs.(tally)))
        plot!(log10.(tally2))
        #disp(log10.(tally2[2500]))
        #=for i in 1:n
            plot!(log10.(tallyv[i,:]))
        end=#
    end
    return result
end

function solvesampTrue(A, v0, z0)
    v = v0
    z = z0
    t = 0
    gamma = 2 / (t + 2)
    w, q, λ = LMOtrueGrad(A, v)
    #W = log10.(abs.(λ))
    while dot(q - v, ∇g(v)) / (f(A, v)^2) > 1e-3
        print(t, "-", dot(q - v, ∇g(v)) / (f(A, v)^2), " ", (f(A, v)^2), "\n")
        append!(lanc2, log10(abs(Result - f(A, v)^2) / Result))
        t = t + 1
        z = sqrt(1 - gamma) * z + sqrt(gamma) * w * rand(Normal(0, 1))
        v = (1 - gamma) * v + gamma * q

        gamma = 2 / (t + 2)
        w, q, λ = LMOtrueGrad(A, v)
        #W = hcat(W,log10.(abs.(λ)))
    end
    x = sign.(A' * z)
    result = (val=f(A, v)^2, x=x, v=v, z=z)
    return result
end

function TriDiagVec(T, V)
    d = size(T, 1)
    k = length(V)
    U = zeros(d, k)
    for i in 1:k
        TT = T - diagm(vec(fill(V[i], (1, d))))
        disp(TT[1:3, 1:3])
        u = zeros(1, d)
        u[1] = 1
        u[2] = -TT[1, 1] / TT[1, 2]
        for i in 2:d-1
            u[i+1] = -(TT[i, i-1] * u[i-1] + TT[i, i] * u[i]) / TT[i, i+1]
        end
        U[:, i] = u / norm
    end
    return U
end

function eigVecFromLanc(α, β, λ)
    w = zeros(m)
    α = α .- λ
    w[1] = 1
    w[2] = -(α[1]) / β[1]
    #disp(w)
    for i in 2:m-1
        #w[i+1] = ((λ - α[i])*w[i] - β[i-1]*w[i-1])/β[i]
        w[i+1] = -(β[i-1] * w[i-1] + (α[i]) * w[i]) / β[i]
    end
    #w[m] = -β[m-1]*w[m-1]/α[m]
    w = w / norm(w)
    q = B(A, w)
    return w, q
end

function solvesampExperiment(A, v0, z0)
    v = v0
    z = z0
    t = 0
    gamma = 2 / (t + 2)
    w, q, λ = LMOtrueGrad(A, v)
    eigValConverge = false
    #disp(λ)
    lastEigVal = λ
    pass = 0
    while dot(q - v, ∇g(v)) / (f(A, v)^2) > 1e-5
        t = t + 1
        z = sqrt(1 - gamma) * z + sqrt(gamma) * w * rand(Normal(0, 1))
        v = (1 - gamma) * v + gamma * q
        gamma = 2 / (t + 2)
        if eigValConverge == false
            w, q, λ = LMOtrueGrad(A, v)
        else
            α, β, q = lanczosv2(A, v)
            T = diagm(0 => α, 1 => β[1:m-1], -1 => β[1:m-1])
            w2, q2, λ = LMOtrueGrad(A, v)
            w1, q1 = eigVecFromLanc(α, β, lastEigVal)
            w1_ = q * w1
            disp((w1_' * grad(A, v) * w1_) / (w1_' * w1_))
            disp((w2' * grad(A, v) * w2) / (w2' * w2))
            #disp(eigvals(T))
            #disp(eigvals(grad(A,v)))
            disp(lastEigVal)
            disp(λ)
            break
        end
        disp(λ)
        #println(abs(λ - lastEigVal))
        if abs(λ - lastEigVal) < 1e-3 && eigValConverge == false
            pass += 1
            println("converge ", t)
            eigValConverge = true
        elseif eigValConverge == false
            lastEigVal = λ
        end
    end
    println("last ", t)
    x = sign.(A' * z)
    result = (val=f(A, v)^2, x=x, v=v, z=z)
    return result
end

#===========================GEN=DATA============================#
function genSample(A)
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
function exp1()
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A / 2
    global m = size(A, 1)
    global n = size(A, 2)
    p = rand(Normal(0, 1), (m, 1))
    p = p / norm(p)
    X0 = p * p'
    val, vec = eigen(X0)
    val = max.(1e-7, val)
    X0 = Matrix(Hermitian(vec * Diagonal(val) * inv(vec)))
    z = rand(MvNormal(zeros(m), X0))
    v = B_(A, X0)
    disp(@benchmark solvesampPow($A, $v, $z))
end

function exp2()
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A / 2
    global m = size(A, 1)
    global n = size(A, 2)
    #z,v = genSample(A)

    maxepoch = 1000
    maxrep = 1
    tally1 = zeros(0)
    tally2 = zeros(0)
    v0 = zeros(n)
    for epoch in 1:maxepoch
        maxres = 0
        maxstartres = 0
        for rep in 1:maxrep
            z, v = genSample(A)
            maxstartres = max(maxstartres, round(norm(A * sign.(A' * z))^2))
            result = solvesampPow(A, v, z)
            maxres = max(maxres, round(norm(A * result.x)^2))
            #print(round(result.val, digits = 1),'-')
            if epoch == 1
                v0 = result.v
            else
                print(round(sum((v0 - result.v) .^ 2), digits=4), '-')
            end
        end
        append!(tally1, maxres)
        append!(tally2, maxstartres)
    end
    print("\n")
    mvavg1 = zeros(maxepoch)
    mvavg2 = zeros(maxepoch)
    cumsum1 = 0
    cumsum2 = 0
    for i in 1:maxepoch
        cumsum1 = cumsum1 + tally1[i]
        mvavg1[i] = cumsum1 / i
        cumsum2 = cumsum2 + tally2[i]
        mvavg2[i] = cumsum2 / i
    end

    m = mean(tally1)
    s = stdm(tally1, m)
    disp("mean", m)
    disp("stdev", s)
    #gr()

    minval = minimum(tally1)
    maxval = maximum(tally1)
    disp("minval", minval)
    disp("maxval", maxval)
    #histogram(tally1,bins = 7)
    samplespace = minval:1:maxval
    bins = [count(x -> (lb <= x < lb + 1), tally1) for lb in samplespace]
    bar(samplespace, bins)

    xx = collect(LinRange(minval, m + 3 * s, 100))
    #disp("x`",xx .- m)
    yy = vec(maxepoch / (s * sqrt(2 * π)) .* exp.(-1 / 2 .* ((xx .- m) ./ s) .^ 2))
    plot!(xx, yy, lw=5)

    #=
    plot(mvavg1)
    plot!(mvavg2)
    print(maximum(tally1),'\n')
    print(maximum(tally2),'\n')
    print(mvavg1[maxepoch],'\n',mvavg2[maxepoch],'\n')
    =#
    savefig("plotconversion.png")


end

function exp3()
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A / 2
    P = readdlm("P.csv", ',', Float64, '\n')
    global Result = readdlm("result.csv", ',', Float64, '\n')[1]
    global m = size(A, 1)
    global n = size(A, 2)
    #z,v = genSample(A)

    maxepoch = 1
    maxrep = 1
    tally = zeros(0)
    #tally2 = zeros(0)
    v0 = zeros(n)
    for epoch in 1:maxepoch
        maxres = 0
        maxstartres = 0
        for rep in 1:maxrep
            print(epoch, "-")
            z, v = genSample(A)
            maxstartres = max(maxstartres, round(norm(A * sign.(A' * z))^2))
            result = solvesampPow(A, v, z, plotcon=true, ϵ=1e-4)
            maxres = max(maxres, round(norm(A * result.x)^2))
            #disp(result.val/Result)
            append!(tally, abs(Result - result.val) / Result)
            #=
            if epoch == 1
                v0 = result.v
            else
                print(round(sum((v0 - result.v).^2),digits = 4),'-' )
            end
            =#
        end
        #append!(tally1,maxres) 
        #append!(tally2,maxstartres) 
    end
    #histogram!(tally,bins = 0:0.001:0.1)
    #savefig("plot/plotvtrack10000Large.png")
end

function exp4()
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A / 2
    P = readdlm("P.csv", ',', Float64, '\n')
    global Result = readdlm("result.csv", ',', Float64, '\n')[1]
    global m = size(A, 1)
    global n = size(A, 2)
    #z,v = genSample(A)

    maxepoch = 1
    maxrep = 1
    tally = zeros(0)
    v0 = zeros(n)
    for epoch in 1:maxepoch
        maxres = 0
        maxstartres = 0
        for rep in 1:maxrep
            print(epoch, "-")
            z, v = genSample(A)
            maxstartres = max(maxstartres, round(norm(A * sign.(A' * z))^2))
            result = solvesampLanc(A, v, z, plotcon=true, ϵ=1e-4)
            maxres = max(maxres, round(norm(A * result.x)^2))
            append!(tally, abs(Result - result.val) / Result)
            #=
            if epoch == 1
                v0 = result.v
            else
                print(round(sum((v0 - result.v).^2),digits = 4),'-' )
            end
            =#
        end
        #append!(tally1,maxres) 
        #append!(tally2,maxstartres) 
    end
    #histogram!(tally,bins = 0:0.001:0.1)
    #savefig("plot/plotvtrack10000LargeLanc.png")
end

function exp5()
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A / 2
    P = readdlm("P.csv", ',', Float64, '\n')
    global Result = readdlm("result.csv", ',', Float64, '\n')[1]
    global m = size(A, 1)
    global n = size(A, 2)
    #z,v = genSample(A)

    maxepoch = 1
    maxrep = 1
    tally = zeros(0)
    tally2 = zeros(0)
    v0 = zeros(n)
    for epoch in 1:maxepoch
        print(epoch, "-")
        maxres = 0
        maxstartres = 0
        for rep in 1:maxrep
            z, v = genSample(A)
            maxstartres = max(maxstartres, round(norm(A * sign.(A' * z))^2))
            result = solvesampTrue(A, v, z)
            maxres = max(maxres, round(norm(A * result.x)^2))
            append!(tally, abs(Result - result.val) / Result)
            #disp(result.val/Result)
            #=if epoch == 1
                v0 = result.v
            else
                print(round(sum((v0 - result.v).^2),digits = 4),'-' )
            end
            =#
        end
        #append!(tally1,maxres) 
        #append!(tally2,maxstartres) 
    end
    #histogram!(tally,bins = 0:0.0001:0.015)
    #savefig("plot/plotcollage.png")
end
function exp6()
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A / 2
    P = readdlm("P.csv", ',', Float64, '\n')
    global Result = readdlm("result.csv", ',', Float64, '\n')[1]
    global m = size(A, 1)
    global n = size(A, 2)
    #z,v = genSample(A)

    maxepoch = 1
    maxrep = 1
    tally = zeros(0)
    tally2 = zeros(0)
    v0 = zeros(n)
    for epoch in 1:maxepoch
        print(epoch, "-")
        maxres = 0
        maxstartres = 0
        for rep in 1:maxrep
            z, v = genSample(A)
            maxstartres = max(maxstartres, round(norm(A * sign.(A' * z))^2))
            result = solvesampLanc2(A, v, z, plotcon=true, ϵ=1e-4)
            maxres = max(maxres, round(norm(A * result.x)^2))
            append!(tally, abs(Result - result.val) / Result)
            #disp(result.val/Result)
            #=if epoch == 1
                v0 = result.v
            else
                print(round(sum((v0 - result.v).^2),digits = 4),'-' )
            end
            =#
        end
        #append!(tally1,maxres) 
        #append!(tally2,maxstartres) 
    end
    #histogram!(tally,bins = 0:0.0001:0.015)
    #savefig("plot/plotcollage.png")
end


function exp7()
    for i in 1:1000
        lanczosv2(A, v)
    end
end

function exp8(file)
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A / 2
    P = readdlm("P.csv", ',', Float64, '\n')
    global Result = readdlm("result.csv", ',', Float64, '\n')[1]
    global m = size(A, 1)
    global n = size(A, 2)
    z, v = genSample(A)
    v, w_ = solvesampTrue(A, v, z)
    G = grad(A, v)
    e = eigvals(G)
    evec = eigvecs(G)
    #disp(e)
    λ = e[1]
    #disp(e[1])
    #disp(λ)
    ws = rand(Normal(0, 1), (m, 1))
    α, β, Q = lanczosv2(A, v, w=ws)
    val, vec = eigen(diagm(0 => α, 1 => β[1:m-1], -1 => β[1:m-1]))
    w4 = (Q*vec)[:, 1]

    w1 = lanczosExperimental(A, v, λ, w=ws)
    #disp(α - al1)
    w2 = eigvecs(G)[:, 1]
    w3, q = LMOtrueGrad(A, v)
    disp(w1' * G * w1 / (w1' * w1))
    disp(w2' * G * w2 / (w2' * w2))
    disp(w3' * G * w3 / (w3' * w3))
    disp(w4' * G * w4 / (w4' * w4))
    tally = zeros(0)
    for i in (e[1]-0.1):0.001:0
        #print(i)
        w1 = lanczosExperimental(A, v, i, w=ws)
        append!(tally, w1' * G * w1 / (w1' * w1))
    end
    plot!([e[1], 0], [w2' * G * w2 / (w2' * w2), w2' * G * w2 / (w2' * w2)], legend=false, color=:red, ls=:dash, lw=1)
    for i in 1:length(e)
        plot!([e[i], e[i]], [0, w3' * G * w3 / (w3' * w3)], legend=false, color=:red, ls=:dot, lw=0.5)
    end
    plot!((e[1])-0.1:0.001:0, tally, legend=false, size=(2000, 1500), dpi=1000)
    if file != "none"
        savefig(file)
    end
end

function exp9()
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A / 2
    P = readdlm("P.csv", ',', Float64, '\n')
    global Result = readdlm("result.csv", ',', Float64, '\n')[1]
    global m = size(A, 1)
    global n = size(A, 2)
    z, v = genSample(A)

    v = solvesampTrue(A, v, z)
    plot(lanc2)
    CSV.write("log8.csv", Tables.table(reshape(lanc2, (1, :))), append=true)
end
exp9()