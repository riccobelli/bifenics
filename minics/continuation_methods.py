# Module collecting several class, each being a general continuation method
from dolfin import plot, FunctionSpace, Function, TestFunction
import matplotlib.pyplot as plt


class ParameterContinuation(object):

    def __init__(self, problem):
        self.problem = problem

    def run(self):
        mesh = self.problem.mesh()
        V = self.problem.function_space(mesh)
        params = self.problem.parameters()
        u = Function(V)
        v = TestFunction(V)
        F = self.problem.residual(u, v, params)

        plt.show()
