using LinearAlgebra
using Distributions
using Plots
using BenchmarkTools
using Statistics
using DelimitedFiles

function g(A,P)
    d = size(A,1)
    n = size(A,2)
    r = zeros(d,d)
    for i in 1:n
        a = A[:,i]
        r = r - (a'*P'*a)^-0.5/2*a*a'
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

function solveFW(A)
    d = size(A,1)
    bestw = rand(Uniform(-1/sqrt(20), 1/sqrt(20)),(d,1))
    tmp = bestw * bestw'
    tmp = tmp /tr(tmp)
    epoch = 0
    
    tally = zeros(0)
    tally1 = zeros(0)
    while true
        epoch = epoch + 1
        grad = g(A,tmp)
        (eig,eigv) = eigen(grad)
        idx = argmin(eig)
        H = eigv[:,idx]*eigv[:,idx]'
        gamma = 2/(epoch+2)
        
        print(epoch,"-",)
        tt = tmp - H
        append!(tally,tr(tt'*grad))
        append!(tally1,f(A,tmp)^2)
        if epoch >= 10000
            break
        end
        tmp = (1-gamma) * tmp + gamma * H
    end
    plot!((tally1))
    U,S,V = svd(tmp)
    U = U[:,1]
    S = S[1]
    re = sign.(A\U*sqrt(S))
    #print(re)
    bw = A*re/norm(A*re)
    return f(A,bw*bw')^2
end
#=
A = transpose([
        1  1  0  0  0  0;
       -1  0  1  1  0  0;
        0  -1 -1  0  1  0;
        0  0  0 -1  0  1;
        0  0  0  0 -1 -1])
=#
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
=#
A = readdlm("graph.csv", ',', Float64, '\n')
#display(@benchmark solveFW(A))

print(solveFW(A))
savefig("plot/plotfFW_3.png")


