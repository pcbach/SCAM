using SparseArrays
open("toy.txt", "w") do io
    n = 316000
    k = 1
    for i = 1:n
        for j = 1:k
            u = i
            v = (i + j - 1) % n + 1
            println(io, u, " ", v)
        end
    end
end