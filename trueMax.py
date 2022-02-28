import cvxpy as cp
import numpy as np
import scipy
import mosek
import csv
np.set_printoptions(precision = 2, suppress=True)

m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(1)
tau = cp.Variable(1)
k = cp.bmat([[x,tau],[tau,1]])

'''
objective = cp.Maximize(tau)
constraints = [ x == 2, k >> 0]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(x.value)
print(tau.value)
print(k.value)
'''

data = []
with open('graph.csv', 'r') as file:
	reader = csv.reader(file)
	for rows in reader:
		row = [float(item)/2 for item in rows]
		data.append(row)
A = np.array(data)
m,n = A.shape
print(n,m)
constraints = []
v = []
tau = []
P = cp.Variable((m,m))

for i in range(n):
	v.append(cp.Variable(1))
	tau.append(cp.Variable(1))

for i in range(n):
	ai = A[:,i, None]
	k = cp.bmat([[v[i],tau[i]],[tau[i],1]])
	constraints += [v[i] >= 0]
	constraints += [tau[i] >= 0]
	constraints += [v[i] == ai.T @ P @ ai]
	constraints += [cp.bmat([[v[i],tau[i]],[tau[i],1]]) >> 0]

constraints += [P >> 0]
constraints += [cp.trace(P) == 1]
objective = cp.Maximize(sum(tau))
prob = cp.Problem(objective, constraints)
result = prob.solve(solver = cp.MOSEK)

'''
print(P.value)
for i in range(n):
	print(v[i].value)
	print(tau[i].value)
'''
print(result**2)