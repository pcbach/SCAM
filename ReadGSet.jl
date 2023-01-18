using SparseArrays
function readfile(filename)
    file = open(filename)
    m = 0
    n = 0
    for ln in eachline(file)
        m += 1
        splitted = split.(ln, " ")
        i = parse(Int64, splitted[1])
        j = parse(Int64, splitted[2])
        n = maximum([n, i, j])
    end
    A = spzeros(m, n)
    C = spzeros(n, n)
    #println(n, " ", m)
    cnt = 1
    close(file)
    file = open(filename)
    for ln in eachline(file)
        splitted = split.(ln, " ")
        i = parse(Int64, splitted[1])
        j = parse(Int64, splitted[2])
        if (cnt % 10000 == 0)
            println(cnt)
        end
        A[cnt, j] = 1
        A[cnt, i] = -1
        C[i, j] -= 1
        C[j, i] -= 1
        C[i, i] += 1
        C[j, j] += 1
        cnt += 1
    end
    return A, C
end
#display(readfile("data.txt"))