using LinearAlgebra
using Distributions
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
        1  1  0  0  0  0;
        -1 0  1  1  0  0;
        0  -1 -1  0  1  0;
        0  0  0 -1  0  1;
        0  0  0  0 -1 -1])
# Check loop
for i in 0:32
    x = [-1; -1; -1; -1; -1]
    #print(i,'\n')
    for bit in 0:4
        #print(1<<bit & i,'\n')
        if (1<<bit) & i != 0 
            x[bit+1] = 1
        end
    end
    #print(x,'\n')
    print(i,':',norm(A * x)^2,'\n')
end

x = [-1; 1; -1; -1; 1]
best = -1e9
bestw = [-1; 0; 1; 1; -1; -1]/sqrt(5)
for epoch in 1:100000
    #print(seed1,' ',seed2,'\n')
    w = rand(Uniform(-1/sqrt(6), 1/sqrt(6)),(6,1))
    w = w 
    #print(w,'\n')
    #print(A*x,'\n')
    #print(dot(A*x,w),'\n')
    if dot(A*x,w) < best
        global best = dot(A*x,w)
        global bestw = copy(w)
    end
end
print(A*x)
print(bestw)
print(f(A,bestw*bestw'))

