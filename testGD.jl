using DelimitedFiles
using SparseArrays
include("MESDP.jl")
function test()
    x = readdlm("C:/Users/pchib/Desktop/MASTER/MESDP/GoogleDistance/people2.txt") 
    l = size(x,1)
    A = spzeros(Int(l*(l-1)/2),l)

    k = 1
    for i = 1:l
        for j = (i+1):l
            A[k,i] = sqrt(x[i,j])
            A[k,j] = -sqrt(x[i,j])
            k = k+1
        end
    end
    global m = size(A, 1)
    global n = size(A, 2)
    A_s = sparse(A/2)
    p = ones((n, 1))
    p[1] = -1
    p = A_s * p
    p = p / norm(p)
    v = B(A_s, v=p)
    p = nothing
    result = Solve(A_s, v, lowerBound=0, upperBound=1e16, printIter=true, plot=true, linesearch=false)
    disp(result.val)
    #####################################
    
    #####################################
end
test()