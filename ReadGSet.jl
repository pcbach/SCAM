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
    #println(n, " ", m)
    cnt = 1
    close(file)
    file = open(filename)
    for ln in eachline(file)
        splitted = split.(ln, " ")
        i = parse(Int64, splitted[1])
        j = parse(Int64, splitted[2])

        #println(cnt, " ", i, " ", j)
        A[cnt, i] = 1
        A[cnt, j] = -1
        cnt += 1
    end
    return A
end
#display(readfile("data.txt"))