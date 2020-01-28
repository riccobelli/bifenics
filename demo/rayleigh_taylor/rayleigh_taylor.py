# We consider a rectangle, composed of a slightly compressible neo-Hookean
# material and subjected to a body force gamma along the y direction. It
# is inspired by the problem described in
#
# Mora, S., Phou, T., Fromental, J. M., & Pomeau, Y. (2014). "Gravity
# driven instability in elastic solid layers". Physical review letters,
# 113(17), 178301.

from minics import NonlinearProblem, ParameterContinuation
from dolfin import RectangleMesh, Point, VectorFunctionSpace, grad, Identity,\
    inner, derivative, dx, tr, Constant, ln, det, SubDomain, near,\
    DirichletBC, MeshFunction


class RayleighTaylor(NonlinearProblem):

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            TOL = 1e-4
            return on_boundary and near(x[1], 0., TOL)

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            TOL = 1e-4
            return on_boundary and near(x[0], 0., TOL)

    class Right(SubDomain):
        def __init__(self, L):
            self.L = L
            SubDomain.__init__(self)  # Call base class constructor!

        def inside(self, x, on_boundary):
            TOL = 1e-4
            return on_boundary and near(x[0], self.L, TOL)

    def __init__(self, L, H, mu, K, nx=10, ny=10):
        self.L = L
        self.H = H
        self.nx = nx
        self.ny = ny

        # Elastic constants
        self.mu = Constant(mu)
        self.K = Constant(K)

    def mesh(self):
        return RectangleMesh(
            Point((0, 0)), Point((self.L, self.H)), self.nx, self.ny, "left/right")

    def function_space(self, mesh):
        return VectorFunctionSpace(mesh, "CG", 1)

    def parameters(self):
        return {"gamma": Constant(0)}

    def boundary_conditions(self, mesh, V):
        bottom = self.Bottom()
        left = self.Left()
        right = self.Right(self.L)

        boundaries = MeshFunction("size_t", mesh, 1)
        boundaries.set_all(0)
        bottom.mark(boundaries, 1)
        left.mark(boundaries, 2)
        right.mark(boundaries, 3)

        bcb = DirichletBC(V, Constant((0, 0)), boundaries, 1)
        bcl = DirichletBC(V.sub(0), Constant(0), boundaries, 2)
        bcr = DirichletBC(V.sub(0), Constant(0), boundaries, 3)

        return [bcb, bcl, bcr]

    def residual(self, u, v, gamma):
        F = Identity(2) + grad(u)
        C = F.T * F
        J = det(F)
        W = self.mu * (tr(C) - 2 * ln(J)) + self.K * ln(J) * ln(J)

        ext_forces = gamma * inner(u, Constant((0, 1)))
        psi = (W - ext_forces) * dx

        return derivative(psi, u, v)

    def solver_parameters(self):
        parameters = {
            'nonlinear_solver': 'snes',
            'snes_solver': {
                'linear_solver': 'mumps',
                'absolute_tolerance': 1e-10,
                'relative_tolerance': 1e-10,
                'maximum_iterations': 10,
            }
        }
        return parameters


if __name__ == '__main__':
    XDMF_options = {"flush_output": True,
                    "functions_share_mesh": True,
                    "rewrite_function_mesh": False}
    rt = RayleighTaylor(1, 1, 1, 100, nx=30, ny=30)
    analysis = ParameterContinuation(
        rt,
        "gamma",
        start=0,
        end=30,
        dt=.05,
        saving_file_parameters=XDMF_options)
    analysis.run()
