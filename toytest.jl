
using TimerOutputs
include("MESDP.jl")
include("ReadGSet.jl")
using Plots
using TickTock
const to = TimerOutput()
function exp3(inputFile, D_sp=0)
    @timeit to "read file" begin
        file = inputFile
        A = readfile(file)
        global m1 = size(A, 1)
        global n = size(A, 2)
        A = A / 2
        #C = A * A'
        A_s = sparse(A)
        global m = size(A_s, 1)
        global n = size(A_s, 2)
        #println(m1, " ", n, " ", m)
        C = A_s' * A_s
        #disp(C)
        v0 = B(A_s, identity=1 / m)

        rows = rowvals(A_s)
        vals = nonzeros(A_s)
        D = spzeros(n)
        for i in 1:n
            for k in nzrange(A_s, i)#
                D[i] += vals[k]^2
            end
        end
        lower = 0
        upper = sqrt(2 * sum(D))
    end

    @timeit to "solve" begin
        result = Solve(A_s, v0, D=D, lowerBound=lower, upperBound=upper, numSample=1, ε=1e-2, timer=to)
    end
    #disp(P)
    #disp(result.t)
    cv = CutValue(A_s, result.z)
    #disp(cv.cut)
    #disp(Vector(cv.cut), name="Cut")
    disp(result.val^2, name="Primal objective")
    disp(cv.val, name="Cut Value")
end


pyplot();
exp3("toy.txt")
#savefig("Result/exp3/toy.png")
show(to)