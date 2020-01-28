from dolfin import Function


class NonlinearProblem(object):
    """
    Generates a new object describing a nonlinear problem
    """

    def mesh(self):
        return NotImplementedError

    def function_space(self, mesh):
        return NotImplementedError

    def residual(self, u):
        return NotImplementedError

    def solver_parameters(self):
        return {}

    def monitor(self, solution, param):
        pass

    def initial_guess(self, V):
        return Function(V)
