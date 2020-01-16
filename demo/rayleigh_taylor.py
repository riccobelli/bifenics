from minics import NonlinearProblem, ParameterContinuation
from dolfin import RectangleMesh, Point, VectorFunctionSpace, grad, Identity,\
    derivative, dx, tr, Constant
import numpy

class RayleighTaylor(NonlinearProblem):

    def __init__(self, L, H):
        self.L = L
        self.H = H

    def mesh(self):
        return RectangleMesh(Point((0, 0)), Point((self.L, self.H)), 5, 5)

    def function_space(self, mesh):
        return VectorFunctionSpace(mesh, "CG", 1)

    def parameter(self):
        return {"gamma" = Constant(0)}

    def residual(self, u, v, g):
        F = Identity(2) + grad(u)
        C = F.T*F
        I1 = tr()


rt = RayleighTaylor(1, 1)
analysis = ParameterContinuation(rt)
analysis.run()
