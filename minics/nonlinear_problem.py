class NonlinearProblem(object):
    def __init__(self):
        '''
        Generates a new object describing a nonlinear_problem
        '''

    def mesh(self, comm):
        return NotImplementedError

    def function_space(self, mesh):
        return NotImplementedError

    def residual(self, u):
        return NotImplementedError
