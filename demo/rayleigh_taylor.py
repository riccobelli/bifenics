from minics import NonlinearProblem, ParameterContinuation
from dolfin import RectangleMesh, Point, VectorFunctionSpace, grad, Identity,\
    derivative, dx, tr, Constant, ln, det
import numpy


class RayleighTaylor(NonlinearProblem):

    def __init__(self, L, H):
        self.L = L
        self.H = H

        # Elastic constants
        self.mu = Constant(1)
        self.K = Constant(10)

    def mesh(self):
        return RectangleMesh(Point((0, 0)), Point((self.L, self.H)), 10, 10)

    def function_space(self, mesh):
        return VectorFunctionSpace(mesh, "CG", 1)

    def parameters(self):
        return {"gamma": Constant(0)}

    def residual(self, u, v, g):
        F = Identity(2) + grad(u)
        C = F.T*F
        J = det(F)
        W = self.mu*(tr(C) - 2*ln(J)) + self.k*ln(J)*ln(J)

        psi = W*dx

        return derivative(psi, u, v)


rt = RayleighTaylor(1, 1)
analysis = ParameterContinuation(rt)
analysis.run()
