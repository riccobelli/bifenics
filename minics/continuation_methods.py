# Module collecting several class, each being a general continuation method
from dolfin import plot, FunctionSpace, Function, TestFunction
import matplotlib.pyplot as plt


class ParameterContinuation(object):

    def __init__(self, problem, param_name, param_start, param_end, dt):
        self.problem = problem
        self._param_name = param_name
        self._param_start = param_start
        self._param_end = param_end
        self._dt = dt

    def run(self):
        # Setting mesh and defining functional spaces edqwop eqowkel epoqwke eqpowk eqpowke
        mesh = self.problem.mesh()
        V = self.problem.function_space(mesh)
        u = Function(V)
        v = TestFunction(V)

        # Setting parameters values
        self.param = self.problem.parameters()[self.param_name]
        self.param.assign(self._param_start)

        F = self.problem.residual(u, v, params)

        plt.show()
