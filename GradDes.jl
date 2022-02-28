using LinearAlgebra
using Distributions
function g(A,P)
    n = size(A,2)
    r = zeros(6,6)
    for i in 1:n
        a = A[:,i]
        r = r - (a'*P'*a)^-0.5/2*a*a'
    end 
    return r
end

function g2(A,w)
    #print(w)
    n_ = size(A,2)
    n = size(A,1)
    r = zeros(n,1)
    for i in 1:n_
        a = Float64.(copy(A[:,i]))
        for j in 1:n
            v = zeros(n,1)
            v[j] = 1
            #print(typeof(a)," ",typeof(w)," ",typeof(v))
            f1 = w'*a*a'*v/sqrt(w'*a*a'*w)
            f2 = w'*a*a'*v/sqrt(a'*w*w'*a)
            #print("\n",f1[1,1]," ",f2[1,1],"\n" )
            print("\n",w'*a*a'*v," ",sqrt(w'*a)*sqrt(a'*w)," ",sqrt(a'*w)*sqrt(w'*a),"\n" )
            r[j] = r[j] - 0.5*( f1[1,1] + f2[1,1] )
        end
    end
    return r
end

function f(A,P)
    n = size(A,2)
    r = 0
    for i in 1:n
        a = A[:,i]
        #print("\n grad ", grad(a,P))
        #print(a,'\n')
        #print(a'*P*a,'\n')
        r = r - sqrt(a'*P*a)
    end 
    return r
end

A = transpose([
        1  0  0  0  0  0;
        -1 0  1  1  0  0;
        0  0 -1  0  1  0;
        0  0  0 -1  0  1;
        0  0  0  0 -1 -1])

bestw = rand(Uniform(-1/sqrt(6), 1/sqrt(6)),(6,1))
tmp = bestw * bestw'
#print(bestw)
#print(tr(tmp))
#display(svd(tmp))
best = f(A,tmp)
gamma = 1e-5
print(g2(A,bestw))
while true
    grad = g2(A,bestw)
    #print(grad)
    #eig = eigvals(grad)
    #eigv = eigvecs(grad)
    #idx = argmin(eig)
    #print(eig[idx])
    #print(eigv[:,idx])
    #H = eigv[:,idx]*eigv[:,idx]'
    #print(H)
    #display(grad)
    #print(tr(tmp),'\n')
    chk = bestw - gamma * grad
    tmp = chk * chk'
    if tr(tmp) > 1
        global gamma = gamma/1.5
    else
        global gamma = 1e-3
        global bestw = bestw - gamma * grad
    end
    if gamma < 1e-5
        break
    end
    print("\n----------------------\n")
    print(eigvals(tmp))
    print("\n----------------------\n")
    #print(f(A,tmp),'\n')
end
tmp = bestw * bestw'
print('\n',best,"  ",f(A,tmp),'\n')
#print('\n',gamma)
#display(tmp)
#display(svd(tmp))

U = svd(tmp).U[:,1]
S = svd(tmp).S[1]
display(U*sqrt(S))

display(sign.(A\U*sqrt(S)))
