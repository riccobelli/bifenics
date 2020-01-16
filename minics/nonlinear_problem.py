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
