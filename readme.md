//2018/07/08 by zeng

This is the 2nd version of the same problem as in master branch. But in this version, I use discrete adjoint method, i.e. I get the dual matrix by transposing the primal matrix rather than directly assemble the dual matrix.

Specificly, in master branch, I follow the following steps to solve the dual problem:
1. derive the dual equation(continuous partial differential equation)
2. discrete it, i.e. assemble the dual matrix and the rhs of dual problem 
3. solve the discret dual equation. 

while in develop branch, the procedure is as follows:
1. assemble the "primal matrix" in a richer space
2. transpose the "primal matrix" in step1 to get the adjoint matrix; assemble the rhs of dual problem
3. solve the discret dual equation.

In literature, the previous one is called "continuous adjoint method", the later one is called "discret adjoint method".
