import pulp
import numpy as np

A=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,2,1,0,0,2,1,0,0], [0,0,0,0,0,0,0,0,1,0,0,0,1], [0,0,0,0,0,0,0,2,0,0,0,2,0], [0,0,0,0,0,0,0,-1,0,0,0,-1,0], [0,-2,1,0,0,0,0,0,0,2,1,0,0], [0,0,0,0,-1,0,0,0,0,0,0,0,1], [0,0,0,-2,0,0,0,0,0,0,0,2,0], [0,0,0,-1,0,0,0,0,0,0,0,-1,0], [0,-2,1,0,0,-2,1,0,0,0,0,0,0], [0,0,0,0,-1,0,0,0,-1,0,0,0,0], [0,0,0,-2,0,0,0,-2,0,0,0,0,0], [0,0,0,-1,0,0,0,-1,0,0,0,0,0]])/6.
print('A', A)

F=[[1,0,0,0,0,0,0,0,0,0,0,0,0], [-1,1,1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,1,1,0,0,0,0,0,0,0,0], [-1,0,0,0,0,1,1,0,0,0,0,0,0], [-1,0,0,0,0,0,0,1,1,0,0,0,0], [-1,0,0,0,0,0,0,0,0,1,1,0,0], [-1,0,0,0,0,0,0,0,0,0,0,1,1]]
print('F', F)
f=np.transpose([1,0,0,0,0,0,0])
print('f', f)
 
E=[[1,0,0,0,0,0,0,0,0,0,0,0,0], [-1,1,1,0,0,0,0,0,0,0,0,0,0], [0,0,-1,1,1,0,0,0,0,0,0,0,0], [-1,0,0,0,0,1,1,0,0,0,0,0,0], [0,0,0,0,0,0,-1,1,1,0,0,0,0], [-1,0,0,0,0,0,0,0,0,1,1,0,0], [0,0,0,0,0,0,0,0,0,0,-1,1,1]]
print('E', E)
e=np.transpose([1,0,0,0,0,0,0])
print('e', e)


# %get dimensions 
# dim_E = size(E)
# dim_F = size(F)
 
# %extend to cover both y and p
# e_new = [zeros(dim_F(2),1);e]
 
# %constraint changes for 2 variables
# H1=[-F,zeros(dim_F(1),dim_E(1))]
# H2=[A,-E']
# H3=zeros(dim_E(2),1)
 
# %bounds for both 
# lb = [zeros(dim_F(2), 1);-inf*ones(dim_E(1),1)]
# ub = [ones(dim_F(2), 1);inf*ones(dim_E(1),1)]
 
# %solve lp problem 
# [yp,fval,exitflag,output,lambda]=linprog(e_new,H2,H3,H1,-f,lb,ub);

#f, A, b, Aeq, beq, lb, ub
#solves min f'*x such that A*X <= b
# Aeq*x = beq
# lb <= x <= ub

#x, fval, exitflag, output, lambda
# fval = f'*x
# lambda contains the Lagrange multipliers at the solution x
model = pulp.LpProblem("Kuhn", pulp.LpMinimize)

p = pulp.LpVariable('p')
y = pulp.LpVariable('y', 0)

model += np.transpose(e)*p
model += -A*y + np.transpose(E)*p >= 0
model += -F * y == -f

model.solve()
model.LpStatus[model.status]

print(p.varValue)
print(y.varValue)
print(pulp.value(model.objective))

# prob.writeLP("Kuhn.lp")
# prob.solve()
# print("Status:", LpStatus[prob.status])

# for v in prob.variables():
# 	print(v.name, "=", v.varValue)

# print("Result = ", value(prob.objective))

 
# %get solutions {x, y, p, q} 
# x = lambda.ineqlin
# y = yp(1 : dim_F(2)) 
# p = yp(dim_F(2)+1 : dim_F(2)+dim_E(1)) 
# q = lambda.eqlin



