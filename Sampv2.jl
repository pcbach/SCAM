using LinearAlgebra
using Distributions
using Plots
using BenchmarkTools
using Statistics
using Arpack 
using DelimitedFiles
using DataStructures
#using PlotlyJS

n = 0 #vertex
m = 0 #edge
Result = 0 #edge

#==============================MISC=============================#
function disp(quan;name = "")
    if name != ""
        print(name,":\n")
    end
    display(quan)
    print('\n')
end
#==============================GRAPH============================#
function initGraphAdj(edgeList)
    Adj = Any[]
    for i in 1:n
        push!(Adj,[])
    end
    for i in 1:m
        v1 = minimum(edgeList[i,:])
        v2 = maximum(edgeList[i,:])
        push!(Adj[v1],i)
        push!(Adj[v2],-i)
    end
    return Adj
end

function initGraphA(edgeList)
    A = zeros(m,n)
    for i in 1:m
        u = edgeList[i,1]
        v = edgeList[i,2]
        A[i,u] = 1
        A[i,v] = -1
    end
    return A
end
#==============================ALGORITHM========================#
function grad_(A,P)
    r = zeros(m,m)
    for i in 1:n
        a = A[:,i]
        r = r - (a'*P'*a)^-0.5/2*a*a'
    end 
    return r
end

function grad(A,v)
    r = zeros(m,m)
    for i in 1:n
        a = A[:,i]
        r = r - a*a'/sqrt(v[i])
    end 
    return r
end

function B(A,w)
    r = zeros(n)
    for i in 1:n
        a = A[:,i]
        r[i] = (a'*w)[1,1]^2
    end 
    return r
end

function B_(A,P)
    r = zeros(n)
    for i in 1:n
        a = A[:,i]
        r[i] = a'*P'*a
    end 
    return r
end

function ∇g(v)
    return 1 ./ (2 .* ( .√(v)))
end

function Badj(A,w)
    r = zeros(m,m)
    for i in 1:n
        a = A[:,i]
        r = r + w[i]*a*a'
    end 
    return r
end

function LMOFW(A,P)
    ∇P = zeros(m,m)
    for i in 1:n
        a = A[:,i]
        ∇P = ∇P - (a'*P*a)^-0.5/2*a*a'
    end 
    (eig,eigv) = eigen(∇P)
    w = eigv[:,1]
    w = w/norm(w)
    q = B(A,w)
    return w
end

function LMOtrueGrad(A,v)
    ∇P = zeros(m,m)
    for i in 1:n
        a = A[:,i]
        ∇P = ∇P - a*a'/(2*sqrt(v[i]))
    end 
    (eig,eigv) = eigen(∇P)
    w = eigv[:,1]
    w = w/norm(w)
    q = B(A,w)
    return w,q
end

function LMOPI(A,v,ϵ::Float64=1e-3)
    b = rand(Uniform(-1, 1),(m,1))
    b = b/norm(b)
    w = zeros(m)
    epoch = 0
    while true
        w = zeros(m)
        for i in 1:n
            a = A[:,i]
            c1 = -1/(2*sqrt(v[i]))
            w_ = a' * b
            w_ = c1 * a * w_
            w = w + w_
        end
        w = w/norm(w)
        if ((1-abs(dot(b,w))) < ϵ)
            break
        end
        b = copy(w)
        epoch = epoch + 1
    end
    w = w/norm(w)
    q = B(A,w)
    return w,q
end

function f_(A,P)
    n = size(A,2)
    r = 0
    for i in 1:n
        a = A[:,i]
        r = r - sqrt(a'*P*a)
    end 
    return r
end

function f(A,v)
    n = size(A,2)
    r = 0
    for i in 1:n
        r = r - sqrt(v[i])
    end 
    return r
end

function solveFW(A,X0;debug = false)
    X = X0 / tr(X0)
    epoch = 0
    while true
        epoch = epoch + 1
        w = LMOFW(A,X)
        H = w*w'
        gamma = 2/(epoch+2)
        G = grad_(A,X)
        if debug
            print(epoch,": ",f_(A,X)^2,' ', dot(X - H,G),'\n')
        end
        if abs(dot(X - H,G)) < 1e-3
            break
        end
        X = (1-gamma) * X + gamma * H
    end
    U,S,V = svd(X)
    y = U[:,1]*sqrt(S[1])
    x = sign.(A'*y)
    return f_(A,X)^2,x
end

function solvesampPow(A,v0,z0;debugw = false,plotcon = false,ϵ::Float64 = 1e-3,terminate = false,ϵ2::Float64 = 1e-3)
    v = v0
    z = z0
    t = 0
    gamma = 2/(t+2)
    w,q = LMOPI(A,v)
    if plotcon
        tally = zeros(0)
        tally2 = zeros(0)
        tallyv = Array(sort(v))
    end
    #disp(tallyv)
    if debugw
        disp(w,name = "w")
    end
    #print((t,mvavg,buffersize,ϵ))
    while (terminate && t < 1000) || dot(q-v,∇g(v)) > ϵ
        error = dot(q-v,∇g(v))
        if plotcon
            append!(tally,error)
            append!(tally2,(Result - f(A,v)^2)/Result)
        end
        t = t + 1
        z = sqrt(1-gamma)*z + sqrt(gamma)*w*rand(Normal(0,1))
        v = (1-gamma)*v + gamma*q
        tallyv = hcat(tallyv,sort(v))
        
        gamma = 2/(t + 2)
        w,q = LMOPI(A,v,ϵ2)
        if debugw
            disp(w,name = "w")
        end
    end
    
    #disp(tallyv)
    x = sign.(A'*z)
    result = (val = f(A,v)^2,x = x,v = v,z = z)
    if plotcon
        #plot!(log10.(abs.(tally1)))
        #plot!(log10.(tally2))
        #disp(log10.(tally2[2500]))
        for i in 1:n
            plot!(log10.(tallyv[i,:]))
        end
    end
    return result
end

function solvesampTrue(A,v0,z0;debug = false)
    v = v0
    z = z0
    t = 0
    gamma = 2/(t+2)
    w,q = LMOtrueGrad(A,v)
    if debug
        disp("w",w)
    end
    while abs(dot(v-q,∇g(v))) > 1e-3
        #print(t,'-')
        t = t + 1
        z = sqrt(1-gamma)*z + sqrt(gamma)*w*rand(Normal(0,1))
        v = (1-gamma)*v + gamma*q
        
        gamma = 2/(t + 2)
        w,q = LMOtrueGrad(A,v)
        if debug
            disp("w",w)
            #print(t,": ",f(A,v)^2,' ', dot(v-q,∇g(v)),'\n')
        end
    end
    x = sign.(A'*z)
    return f(A,v)^2,x
end

function genSample(A)
    p = rand(Uniform(-1, 1),(m,1))
    p = p
    X0 = p*p'
    X0 = X0/tr(X0)
    val,vec = eigen(X0)
    val = max.(1e-7,val)
    X0 = Matrix(Hermitian(vec*Diagonal(val)*inv(vec)))
    z = rand(MvNormal(zeros(m),X0))
    v = B_(A,X0)
    return z,v
end

function genSampleP(A,P)
    X0 = P/tr(P)
    val,vec = eigen(X0)
    val = max.(1e-10,val)
    X0 = Matrix(Hermitian(vec*Diagonal(val)*inv(vec)))
    z = rand(MvNormal(zeros(m),X0))
    v = B_(A,X0)
    return z,v
end

#==============================MAIN=============================#
function exp1()
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A/2
    global m = size(A,1)
    global n = size(A,2)
    p = rand(Normal(0,1),(m,1))
    p = p/norm(p)
    X0 = p*p'
    val,vec = eigen(X0)
    val = max.(1e-7,val)
    X0 = Matrix(Hermitian(vec*Diagonal(val)*inv(vec)))
    z = rand(MvNormal(zeros(m),X0))
    v = B_(A,X0)
    disp(@benchmark solvesampPow($A,$v,$z))
end

function exp2()
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A/2
    global m = size(A,1)
    global n = size(A,2)
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
            z,v = genSample(A)
            maxstartres = max(maxstartres, round(norm(A*sign.(A'*z))^2))
            result = solvesampPow(A,v,z)
            maxres = max(maxres, round(norm(A*result.x)^2))
            #print(round(result.val, digits = 1),'-')
            if epoch == 1
                v0 = result.v
            else
                print(round(sum((v0 - result.v).^2),digits = 4),'-' )
            end
        end
        append!(tally1,maxres) 
        append!(tally2,maxstartres) 
    end
    print("\n")
    mvavg1 = zeros(maxepoch)
    mvavg2 = zeros(maxepoch)
    cumsum1 = 0
    cumsum2 = 0
    for i in 1:maxepoch
        cumsum1 = cumsum1 + tally1[i]
        mvavg1[i] = cumsum1/i
        cumsum2 = cumsum2 + tally2[i]
        mvavg2[i] = cumsum2/i
    end

    m = mean(tally1)
    s = stdm(tally1,m)
    disp("mean",m)
    disp("stdev",s)
    #gr()

    minval = minimum(tally1)
    maxval = maximum(tally1)
    disp("minval",minval)
    disp("maxval",maxval)
    #histogram(tally1,bins = 7)
    samplespace = minval:1:maxval
    bins = [ count(x->(lb<=x<lb+1),tally1) for lb in samplespace ]
    bar(samplespace,bins)

    xx = collect(LinRange(minval, m+3*s, 100))
    #disp("x`",xx .- m)
    yy = vec(maxepoch/(s*sqrt(2*π)) .* exp.(-1/2 .* ((xx .- m) ./ s) .^ 2))
    plot!(xx,yy,lw = 5)
    
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
    A = A/2
    P = readdlm("P.csv", ',', Float64, '\n')
    global Result = readdlm("result.csv", ',', Float64, '\n')[1]
    global m = size(A,1)
    global n = size(A,2)
    #z,v = genSample(A)

    maxepoch = 1
    maxrep = 1
    tally1 = zeros(0)
    tally2 = zeros(0)
    v0 = zeros(n)
    for epoch in 1:maxepoch
        maxres = 0
        maxstartres = 0
        for rep in 1:maxrep
            z,v = genSampleP(A,P)
            maxstartres = max(maxstartres, round(norm(A*sign.(A'*z))^2))
            result = solvesampPow(A,v,z,plotcon = true,ϵ=1e-3,terminate = true)
            maxres = max(maxres, round(norm(A*result.x)^2))
            disp(result.val)
            if epoch == 1
                v0 = result.v
            else
                print(round(sum((v0 - result.v).^2),digits = 4),'-' )
            end
        end
        append!(tally1,maxres) 
        append!(tally2,maxstartres) 
    end
    savefig("plot/plotvtrack1000Large.png")
end

exp3()
