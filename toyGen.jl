using SparseArrays
open("toy.txt", "w") do io
    n = 1000000
    k = 5
    check = zeros(1, n)
    for i = 1:n
        repeat = zeros(0)
        for j = check[i]+1:k
            u = i
            v = rand(i:n)
            while (check[v] >= k || v == u || v in repeat)
                v = rand(i:n)
            end
            println(io, u, " ", v)
            repeat = append!(repeat, v)
            check[v] = check[v] + 1
            check[u] = check[u] + 1
        end
    end
end