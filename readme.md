//2018/6/22 by zeng

I modify step_12 of deal.ii to solve a 2D linear advection equation. it used adjoint based error estimation method to adaptively refine the mesh. Look at tutorial of step_12 and Ralf Hartmann's doctoral dissertation "Adaptive finite element methods for the compressible euler equations" for more detail.

This master branch used the so-called continuous adjoint method, switch to the develop branch for the discrete adjoint method.
