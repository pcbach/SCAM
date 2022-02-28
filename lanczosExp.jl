using LinearAlgebra
using Distributions
using Plots
using BenchmarkTools
using Statistics
using Arpack 
using DelimitedFiles
using DataStructures

n = 0 #vertex
m = 0 #edg
tallycount = zeros(15)
tallysum = zeros(15)
#==============================MISC=============================#
function disp(quan;name = "")
    if name != ""
        print(name,":\n")
    end
    display(quan)
    print('\n')
end
function grad(A,v)
    r = zeros(m,m)
    for i in 1:n
        a = A[:,i]
        r = r - a*a'/(2*sqrt(v[i]))
    end 
    return r
end

function lanczosSubRoutine(A,v,b)
    wp = zeros(m)
    for i in 1:n
        a = A[:,i]
        c = -1/(2*sqrt(v[i]))
        w_ = a' * b
        w_ = c * a * w_
        wp = wp + w_
    end
    return wp
end

function lanczos(A,v,k)
    Q = zeros(m,k)
    Al = zeros(k)
    B = zeros(k)
    W = zeros(m,k)

    q = rand(Uniform(-1, 1),(m,1))
    q = q/norm(q)

    Q[:,1] = q
    W[:,1] = lanczosSubRoutine(A,v,Q[:,1])
    Al[1] = Q[:,1]' * W[:,1]
    W[:,1] = W[:,1] - Al[1] * Q[:,1]
    T = zeros(k,k)
    T[1,1] = Al[1]
    Result = Q[:,1]/norm(Q[:,1])
    for i in 2:k
        B[i] = norm(W[:,i-1])
        if B[i] == 0
            break
        else
            Q[:,i] = W[:,i-1]/B[i]
        end
        W[:,i] = lanczosSubRoutine(A,v,Q[:,i])
        Al[i] = Q[:,i]' * W[:,i]
        W[:,i] = W[:,i] - Al[i] * Q[:,i] - B[i] * Q[:,i-1]
        T[i,i] = Al[i]
        T[i-1,i] = B[i]
        T[i,i-1] = B[i]

        val,vec = eigen(T[1:i,1:i])
        j = 1
        while j < i
            #print(abs(val[j] - val[1])/val[1],"\n")
            if abs(val[j] - val[1])/abs(val[1]) > 1/10
                break
            end
            j += 1
        end
        print(val[1],"\n",val[j],"\n",val[i],"\n",(val[1] - val[j])/(val[1]-val[i]),"\n")
        NewResult = Q[:,1:i]*vec[:,1]
        NewResult = NewResult/norm(NewResult)
        error = abs(1-abs(dot(Result,NewResult)))
        print("\n---------\n")
        for k in 15:-1:1
            if error < 10.0^(-k)
                
                global tallycount[k] += 1
                global tallysum[k] += i
                break
            end
        end
        Result = NewResult
    end
end

function B_(A,P)
    r = zeros(n)
    for i in 1:n
        a = A[:,i]
        r[i] = a'*P'*a
    end 
    return r
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

function main()
    A = readdlm("graph.csv", ',', Float64, '\n')
    A = A/2
    global m = size(A,1)
    global n = size(A,2)

    z,v = genSample(A)

    q = rand(Uniform(-1, 1),(m,1))
    q = q/norm(q)

    #display(grad(A,v)*q)
    #print("\n")
    #display(lanczosSubRoutine(A,v,q))
    #print(norm(eigvecs(grad(A,v))[:,1]))
    tally = zeros(0)
    #for i = 1:1000
    #    print(i,'-')
    lanczos(A,v,m)
    #end
    #disp(tallysum./tallycount)
    writedlm("tally.csv",  tallysum./tallycount, ',')
    #eigvec(grad(A,v))[:,1])
    #print("\n----------------------------------------\n")
end

main()