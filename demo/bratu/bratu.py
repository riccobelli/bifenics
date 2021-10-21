from bifenics import BifenicsProblem, ParameterContinuation, ArclengthContinuation
from dolfin import (
    IntervalMesh,
    Constant,
    exp,
    inner,
    grad,
    dx,
    DirichletBC,
    FunctionSpace,
    parameters,
)
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

parameters["allow_extrapolation"] = True


class Bratu(BifenicsProblem):
    def __init__(self):
        self.lmbdaplot = np.array([])
        self.uplot = np.array([])

    def mesh(self):
        return IntervalMesh(1000, 0, 1)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        return {"lambda": Constant(0)}

    def residual(self, u, v, parameters):
        lmbda = parameters["lambda"]
        F = -inner(grad(u), grad(v)) * dx + lmbda * exp(u) * v * dx
        return F

    def solver_parameters(self):
        parameters = {
            "nonlinear_solver": "snes",
            "snes_solver": {
                "linear_solver": "mumps",
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 10,
            },
        }

        return parameters

    def boundary_conditions(self, mesh, V):
        return DirichletBC(V, Constant(0), "on_boundary")

    def ac_monitor(self, u, lmbda, parameters, output_file):
        # u_linf = norm(u.vector(), "linf")
        # self.uplot = np.append(self.uplot, u_linf)

        self.uplot = np.append(self.uplot, u(0.5))
        self.lmbdaplot = np.append(self.lmbdaplot, float(lmbda))


if __name__ == "__main__":
    bratu = Bratu()
    XDMF_options = {
        "flush_output": True,
        "functions_share_mesh": True,
        "rewrite_function_mesh": False,
    }
    analysis = ArclengthContinuation(
        bratu,
        "lambda",
        start=0,
        end=3.6,
        ds=0.1,
        saving_file_parameters=XDMF_options,
        first_step_with_parameter_continuation=False,
    )
    analysis.run()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        plt.plot(bratu.lmbdaplot, bratu.uplot, "-o")
        plt.show()
