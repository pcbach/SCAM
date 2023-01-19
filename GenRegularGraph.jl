function Gen(filename, n, d)

    open(filename, "w") do io
        #group i contain i,i+n,i+2n,...i+dn 
        check = zeros((n * d, 1))
        edges = zeros((n, d))
        count = fill(1, (n * d, 1))
        cnt = 0
        while sum(check) < n * d
            u = rand(1:n*d)
            while (check[u] == 1)
                u = rand(1:n*d)
            end

            v = rand(1:n*d)
            while (check[v] == 1)
                v = rand(1:n*d)
            end

            uu = u % n + 1
            vv = v % n + 1
            cnt += 1
            if uu != vv && !(uu in edges[vv, :]) && !(vv in edges[uu, :])
                edges[count[v]] = uu
                count[vv] += 1
                edges[count[uu]] = vv
                count[uu] += 1
                check[v] = 1
                check[u] = 1
                cnt = 0
                println(io, vv, " ", uu)
                if (sum(check) % 10000 == 0)
                    println(sum(check))
                end
            end
            if (cnt > 5 * n * d)
                println("failed")
                break
            end
        end
    end
end

Gen("BigExample/1e4n5d2.txt", 10000, 10)