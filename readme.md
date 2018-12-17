//2018/6/22 by zeng

This code solves a 2D linear advection equation on adaptively refined meshes. It's an upwind DG solver and used adjoint based error estimation method to generate refinement indicators. Look at tutorial of step_12 for more detail.
The structure of this code can be treated as a combination of deal.ii's tutorial step-12 and step-14 and is thus a little bit complicated. Accually I don't think its structure is very good, but anyway, it worked.   

This master branch used the so-called continuous adjoint method, switch to the develop branch for the discrete adjoint method.
