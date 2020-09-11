from dolfin import Function, split, inner, dx


class BifenicsProblem(object):
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

    def monitor(self, solution, param, output_file=None):
        pass

    def initial_guess(self, V):
        return Function(V)

    def modify_initial_guess(self, u, param):
        pass

    def ac_constraint(self, ac_state, ac_state_prev, ac_testFunction, ds):
        u, param = split(ac_state)
        u_prev, param_prev = split(ac_state_prev)

        u_testFunction, param_testFunction = split(ac_testFunction)

        F = (param_testFunction * inner(u - u_prev, u - u_prev) * dx +
             param_testFunction * inner(param - param_prev, param - param_prev) * dx -
             param_testFunction * ds**2 * dx)

        return F
