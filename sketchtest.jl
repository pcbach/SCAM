using LinearAlgebra
using Arpack
A = rand(5, 3)
X = A * A'
#display(X)

r = 2
k = 2 * r#2 * r + 1
l = 2 * r#4 * r + 3

Ω = rand(5, k)
Ψ = rand(l, 5)

Y = X * Ω
W = Ψ * X

Q = qr(Y).Q
B = pinv(Ψ * Q) * W
U, S, V = svds(B, nsv=r)[1]
X_ = Q * U * diagm(S) * V'


display(eigvals(X))
display(eigvals(X_))