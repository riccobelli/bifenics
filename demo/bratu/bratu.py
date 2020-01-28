from minics import NonlinearProblem, ParameterContinuation
from dolfin import IntervalMesh, Constant, exp, inner, grad, dx, DirichletBC,\
    FunctionSpace


class Bratu(NonlinearProblem):

    def mesh(self):
        return IntervalMesh(1000, 0, 1)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        return {"lambda": Constant(0)}

    def residual(self, u, v, lmbda):
        F = - inner(grad(u), grad(v)) * dx + lmbda * exp(u) * v * dx
        return F

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

    def boundary_conditions(self, mesh, V):
        return DirichletBC(V, Constant(0), "on_boundary")


if __name__ == '__main__':
    bratu = Bratu()
    XDMF_options = {"flush_output": True,
                    "functions_share_mesh": True,
                    "rewrite_function_mesh": False}
    analysis = ParameterContinuation(
        bratu,
        "lambda",
        start=0,
        end=3.6,
        dt=.01,
        saving_file_parameters=XDMF_options)
    analysis.run()
