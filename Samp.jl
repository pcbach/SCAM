using LinearAlgebra
using Distributions
using Plots
using BenchmarkTools
using Statistics
using Arpack 
using DelimitedFiles
#using PyPlot

function grad(A,P)
    d = size(A,1)
    n = size(A,2)
    r = zeros(d,d)
    for i in 1:n
        a = A[:,i]
        r = r - (a'*P'*a)^-0.5/2*a*a'
        #print((a'*P'*a),'\n')
        #display(r)
        #print('\n')
    end 
    return r
end

function B(A,w)
    n = size(A,2)
    d = size(A,1)
    r = zeros(n,1)
    for i in 1:n
        a = A[:,i]
        r[i] = (a'*w)[1,1]^2
    end 
    return r
end

function ∇g(v)
    return -1 ./ (2 .* ( .√(v)))
end

function Badj(A,w)
    n = size(A,2)
    d = size(A,1)
    r = zeros(d,d)
    for i in 1:n
        a = A[:,i]
        r = r + w[i]*a*a'
    end 
    return r
end

function f(A,P)
    n = size(A,2)
    r = 0
    for i in 1:n
        a = A[:,i]
        r = r - sqrt(a'*P*a)
    end 
    return r
end

function fB(A,w)
    n = size(A,2)
    r = 0
    for i in 1:n
        a = A[:,i]
        r = r - abs((a'*w)[1,1])
    end 
    return r
end

function proj(u,a)
    return (u'*a)/(u'*u)*u
end

function qrmod(X,n)
    Q = X[:,1]/norm(X[:,1])
    for i in 2:n
        Q = hcat(Q,X[:,i])
        for j in 1:(i-1)
            Q[:,i] = Q[:,i] - proj(Q[:,j],X[:,i])    
        end
        Q[:,i] = Q[:,i]/norm(Q[:,i])
    end
    R = zeros(n,n)
    for i in 1:n
        for j in i:n
            R[i,j] = Q[:,i]'*X[:,j]
        end
    end
    return Q,R        
end

function LMO(grad)
    (eig,eigv) = eigen(grad)
    idx = argmin(eig)
    H = eigv[:,idx]*eigv[:,idx]'
    return eigv[:,idx]
end

function LmoRsvdHm(A,w,r)
    n = size(A,2)
    d = size(A,1)
    Zx = zeros(d,r)
    p = rand(Normal(0,1),(d,r))
    for i in 1:n
        a = A[:,i]
        c1 = -1/(2*abs((a'*w)[1,1]))
        Z = a' * p
        Z = c1 * a * Z
        Zx = Zx + Z
    end 
    Q,R = qrmod(Zx,r)
    Yx = zeros(r,d)
    for i in 1:n
        a = A[:,i]
        c1 = -1/(2*abs((a'*w)[1,1]))
        Y = Q' * a
        Y = c1 * Y * a'
        Yx = Yx + Y
    end 
    Uy,S,V = svd(Yx)
    #display(S)
    U = Q * Uy
    V = -V
    if (S[1] > 0)
        return V[:,1],B(A,V[:,1])
    else
        return zeros((d,1)),zeros((d,1))
    end
end

function PIEig(A,w,b,ϵ)
    n = size(A,2)
    m = size(A,1)
    r = zeros(m)
    epoch = 0
    while true
        for i in 1:n
            a = A[:,i]
            c1 = -1/(2*abs((a'*w)[1,1]))
            r_ = a' * b
            r_ = c1 * a * r_
            r = r + r_
        end
        r = r/norm(r)
        #display(r)
        #print('\n',1-abs(dot(b,r)),"\n")
        if ((1-abs(dot(b,r))) < ϵ)
            break
        end
        b = r
        epoch = epoch + 1
    end
    #print(epoch)
    r = r/norm(r)
    return r
end

function LmoHMImprove(A,w,ϵ)
    v = LmoRsvdHm(A,w,1)[1]
    v = PIEig(A,w,v/norm(v),ϵ)
    return v,B(A,v)
end

function LmoTrue(A,w)
    g = grad(A,w*w')
    (eig,eigv) = eigen(g)
    idx = argmin(eig)
    w = eigv[:,idx]
    q = B(A,w)
    w = w/norm(w)
    return w,q
end

function LmoLanc(A,w)
    g = grad(A,w*w')
    (eig,eigv) = eigs(g,nev = 1,which = :LM)
    idx = argmax(abs.(eig))
    w = eigv[:,idx]
    q = B(A,w)
    w = w/norm(w)
    return w,q
end

function updatevar(z,v,w,q,gamma)
    s = rand(Normal(0,1))
    z = sqrt(1-gamma)*z + sqrt(gamma)*w*s
    v = (1-gamma)*v + gamma*q
    return z,v
end

function updatevar1(z,v,w,q,gamma)
    z = sqrt(1-gamma)*z + sqrt(gamma)*w
    v = (1-gamma)*v + gamma*q
    return z,v
end

function solvesamp(A)
    d = size(A,1)
    p = rand(Uniform(-1, 1),(d,1))
    X0 = p*p'
    val,vec = eigen(X0)
    val = max.(1e-5,val)
    X0 = Matrix(Hermitian(vec*Diagonal(val)*inv(vec)))

    z = rand(MvNormal(zeros(d),X0))
    v = B(A,z)
    t = 0
    gamma = 2/(t+2)
    w,q = LmoRsvdHm(A,z,1)
    w1,q1 = LmoTrue(A,z)
    print("________epoch_",t,"______\n")
    display(w)
    print("\n")
    display(w1)
    print("\n")
    print("______________________\n")
    w,q = LmoTrue(A,z)
    #q = B(A,w)

    while dot(v-q,∇g(v)) > 1e-6
        z,v = updatevar(z,v,w,q,gamma)
        t = t + 1
        gamma = 2/(t + 2)
        w,q = LmoRsvdHm(A,z,1)
        w1,q1 = LmoTrue(A,z)
        print("________epoch_",t,"______\n")
        display(w)
        print("\n")
        display(w1)
        print("\n")
        print("______________________\n")
        w,q = LmoTrue(A,z)
        #q = B(A,w)
    end
    #print("number of epoch ",t,"\n")
    result = sign.(A\z)
    bestw = A*result/norm(A*result)
    return f(A,bestw*bestw')^2
end

function solvesampTrue(A,z;debug::Bool = false)
    v = B(A,z)
    t = 0
    gamma = 2/(t+2)
    w,q = LmoLanc(A,z)
    #q = B(A,w)

    if debug == true
        print(z,'\n')
        print(w,'\n')
        print(q,'\n')
        print("----------------\n")
    end

    while dot(v-q,∇g(v)) > 1e-3
        z,v = updatevar(z,v,w,q,gamma)
        t = t + 1
        gamma = 2/(t + 2)
        w,q = LmoLanc(A,z)

        if debug == true
            print(z,'\n')
            print(w,'\n')
            print(q,'\n')
            print("----------------\n")
        end
    end
    result = sign.(A'*z)
    bestw = A*result/norm(A*result)
    return round(f(A,bestw*bestw')^2),t
end

function solvesampRsvdHm(A,z)
    v = B(A,z)
    t = 0
    gamma = 2/(t+2)
    w,q = LmoRsvdHm(A,z,1)
    #w = PIEig(A,z,w)
    #q = B(A,w)

    while dot(v-q,∇g(v)) > 1e-6
        z,v = updatevar(z,v,w,q,gamma)
        t = t + 1
        gamma = 2/(t + 2)
        w,q = LmoRsvdHm(A,z,1)
        #q = B(A,w)
    end
    #print("number of epoch ",t,"\n")
    result = sign.(A'*z)
    bestw = A*result/norm(A*result)
    return f(A,bestw*bestw')^2,t
end

function solvesampRsvdHmImp(A,z,ϵ)
    v = B(A,z)
    t = 0
    gamma = 2/(t+2)
    w,q = LmoHMImprove(A,z,ϵ)
    #print("imp\n")
    #display(w)
    #print("\n")
    #q = B(A,w)

    while dot(v-q,∇g(v)) > 1e-6
        z,v = updatevar(z,v,w,q,gamma)
        t = t + 1
        gamma = 2/(t + 2)
        w,q = LmoHMImprove(A,z,ϵ)
        #print("imp\n")
        #display(w)
        #print("\n")
        #q = B(A,w)
    end
    #print("number of epoch ",t,"\n")
    result = sign.(A'*z)
    bestw = A*result/norm(A*result)
    return f(A,bestw*bestw')^2,t
end

function solvesampPow(A,z,ϵ::Float64=1e-3;debug::Bool = false)
    v = B(A,z)
    t = 0
    gamma = 2/(t+2)
    #w,q = LmoLanc(A,z)
    p = rand(Uniform(-1, 1),(length(z),1))
    w = PIEig(A,z,p/norm(p),ϵ)
    q = B(A,w)
    if debug == true
        print(z,'\n')
        print(w,'\n')
        print(q,'\n')
        print("----------------\n")
    end
    while dot(v-q,∇g(v)) > 1e-3
        z,v = updatevar1(z,v,w,q,gamma)
        t = t + 1
        gamma = 2/(t + 2)
        p = rand(Uniform(-1, 1),(length(z),1))
        w = PIEig(A,z,p/norm(p),ϵ)
        q = B(A,w)

        if debug == true
            print(z,'\n')
            print(w,'\n')
            print(q,'\n')
            print("----------------\n")
        end
    end
    #print("number of epoch ",t,"\n")
    result = sign.(A'*z)
    bestw = A*result/norm(A*result)
    return round(f(A,bestw*bestw')^2),t
end

A = transpose([
    1  1  0  0  0  0;
    -1  0  1  1  0  0;
    0  -1 -1  0  1  0;
    0  0  0 -1  0  1;
    0  0  0  0 -1 -1])

A = [
    1 -1  0  0  0   0  0  0  0  0   0  0  0  0  0;
    0  1 -1  0  0   0  0  0  0  0   0  0  0  0  0;
    0  0  1 -1  0   0  0  0  0  0   0  0  0  0  0;
    0  0  0  1 -1   0  0  0  0  0   0  0  0  0  0;
    0  0  0  0  1  -1  0  0  0  0   0  0  0  0  0;

    0  0  0  0  0   1 -1  0  0  0   0  0  0  0  0;
    0  0  1  0  0   0  0  0  0 -1   0  0  0  0  0;
    0  0  0  1  0   0  0  0  0 -1   0  0  0  0  0;
    0  0  0  1  0   0  0  0  0  0  -1  0  0  0  0;
    0  0  0  0  1   0  0 -1  0  0   0  0  0  0  0;

    0  0  0  0  0   1  0 -1  0  0   0  0  0  0  0;
    0  0  0  0  0   0  0  0  1 -1   0  0  0  0  0;
    0  0  0  0  0   0  0  1  0  0  -1  0  0  0  0;
    0  0  0  0  0   0  0  0  0  1   0 -1  0  0  0;
    0  0  0  0  0   0  0  0  0  1   0  0 -1  0  0;

    0  0  0  0  0   0  0  0  0  0   1  0  0 -1  0;
    0  0  0  0  0   0  0  0  0  0   0  0  1  0 -1;
    0  0  0  0  0   0  0  0  0  0   0  0  0  1 -1;
    0  0  0  0  0   0  0  0  0  0   0  1 -1  0  0;
    0  0  0  0  0   0  0  0  0  0   0  0  1 -1  0;
]

#=
A = [
    1 -1  0  0  0   0  0  0  0  0; 
    0  1 -1  0  0   0  0  0  0  0;  
    1  0  0 -1  0   0  0  0  0  0;  
    1  0  0  0 -1   0  0  0  0  0;  
    0  0  1  0 -1   0  0  0  0  0;  
    
    0  0  1  0  0  -1  0  0  0  0;  
    0  0  0  1  0   0 -1  0  0  0;  
    0  0  0  1 -1   0  0  0  0  0;  
    0  1  0  0  0   0 -1  0  0  0;  
    0  0  0  0  1  -1  0  0  0  0;  
    
    0  0  0  0  1   0  0  0  0 -1;  
    0  0  0  0  0   0  1  0 -1  0;  
    0  0  0  0  0   0  1  0  0 -1;  
    0  0  0  0  0   0  0  0  1 -1;  
    0  0  0  0  0   0  0  1 -1  0;    
]   
=##
#=
A = readdlm("graph.csv", ',', Float64, '\n')
m = size(A,1)
n = size(A,2)
p = rand(Uniform(-1, 1),(m,1))
#p = 1:20
#p = [0.1,0.2,0.3,0.4,0.5,0.6]
X0 = p*p'
val,vec = eigen(X0)
val = max.(1e-5,val)
X0 = Matrix(Hermitian(vec*Diagonal(val)*inv(vec)))
z = rand(MvNormal(zeros(m),X0))
q = rand(Uniform(-1, 1),(m,1))
q = q/norm(q)
#=
display(LmoTrue(A,p)[1])
print('\n')
q = rand(Uniform(-1, 1),(m,1))
display(PIEig(A,p,q,1e-3))
print('\n')
=#
a = LmoLanc(A,p)[1]
b = LmoTrue(A,p)[1]
c = PIEig(A,p,q,1e-4)
print(dot(a,b),'\n')
print(dot(b,c),'\n')
print(dot(c,a),'\n')

#print(solvesampRsvdHm(A))
#print(solvesampTrue(A))
#=
m = size(A,1) #edge
n = size(A,2) #vertex
p = rand(Uniform(-1, 1),(m,1))
X0 = p*p'
val,vec = eigen(X0)
val = max.(1e-5,val)
X0 = Matrix(Hermitian(vec*Diagonal(val)*inv(vec)))
z = rand(MvNormal(zeros(d),X0))
=#
#=
m = size(A,1)
p = rand(Uniform(-1, 1),(m,1))

v0 = LmoTrue(A,p)[1]
v1 = LmoRsvdHm(A,p,1)[1]
v2 = LmoRsvdHm(A,p,2)[1]
v3 = LmoLanc(A,p)[1]
display(v0)
print("\n")
display(v1)
print("\n")
display(v2)
print("\n")
display(v3)
print("\n")
print(dot(v0,v1),'\n')
print(dot(v0,v2),'\n')
print(dot(v0,v3))

display(@benchmark LmoRsvdHm(A,p,1)[1])
display(@benchmark LmoRsvdHm(A,p,2)[1])
display(@benchmark LmoLanc(A,p)[1])
display(@benchmark LmoTrue(A,p)[1])
=#


#print(solvesamp(A))

#=
maxepoch = 500
maxrep = 1
tally1 = zeros(0)
tally2 = zeros(0)
tally3 = zeros(0)
tally4 = zeros(0)
for epoch in 1:maxepoch
    print("\n",epoch,"\n")
    result1 = 0
    result2 = 0
    result3 = 0
    result4 = 0
    for rep in 1:maxrep
        p = rand(Uniform(-1, 1),(m,1))
        X0 = p*p'
        val,vec = eigen(X0)
        val = max.(1e-5,val)
        X0 = Matrix(Hermitian(vec*Diagonal(val)*inv(vec)))
        z = rand(MvNormal(zeros(m),X0))
        re1 = solvesampTrue(A,z)[1]
        re2 = solvesampRsvdHm(A,z)[1]
        re3 = solvesampRsvdHmImp(A,z,1e-1)[1]
        re4 = solvesampPow(A,z,1e-3)[1]
        result1 = max(re1, result1) 
        result2 = max(re2, result2)
        result3 = max(re3, result3)
        result4 = max(re4, result4)
        
    end
    append!(tally1,result1) 
    append!(tally2,result2) 
    append!(tally3,result3) 
    append!(tally4,result4) 
end

mvavg1 = zeros(maxepoch)
mvavg2 = zeros(maxepoch)
mvavg3 = zeros(maxepoch)
mvavg4 = zeros(maxepoch)
cumsum1 = 0
cumsum2 = 0
cumsum3 = 0
cumsum4 = 0
for i in 1:maxepoch
    global cumsum1 = cumsum1 + tally1[i]
    mvavg1[i] = cumsum1/i
    global cumsum2 = cumsum2 + tally2[i]
    mvavg2[i] = cumsum2/i
    global cumsum3 = cumsum3 + tally3[i]
    mvavg3[i] = cumsum3/i
    global cumsum4 = cumsum4 + tally4[i]
    mvavg4[i] = cumsum4/i
end

gr()
plot(mvavg1)
plot!(mvavg2)
plot!(mvavg3)
plot!(mvavg4)

print(mvavg1[maxepoch],'\n',mvavg2[maxepoch],'\n',mvavg3[maxepoch],'\n',mvavg4[maxepoch])
savefig("plot5.png")

p = rand(Uniform(-1, 1),(m,1))
X0 = p*p'
val,vec = eigen(X0)
val = max.(1e-5,val)
X0 = Matrix(Hermitian(vec*Diagonal(val)*inv(vec)))
z = rand(MvNormal(zeros(m),X0))

display(@benchmark solvesampTrue(A,z))
display(@benchmark solvesampRsvdHm(A,z))
display(@benchmark solvesampRsvdHmImp(A,z,1e-1))
display(@benchmark solvesampPow(A,z,1e-3))
=#
=#  