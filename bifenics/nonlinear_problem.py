# Copyright (C) 2023 Davide Riccobelli
#
# This file is part of BIFEniCS library for FEniCS.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2.1 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


# This file implements a base class for bifurcation problems.


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

    def ac_monitor(self, solution, ac_param, parameters, output_file=None):
        pass

    def initial_guess(self, V):
        return Function(V)

    def modify_initial_guess(self, u, param):
        pass

    def ac_constraint(self, ac_state, ac_state_prev, ac_testFunction, ds):
        u, param = split(ac_state)
        u_prev, param_prev = split(ac_state_prev)

        u_testFunction, param_testFunction = split(ac_testFunction)

        F = (
            param_testFunction * inner(u - u_prev, u - u_prev) * dx
            + param_testFunction * inner(param - param_prev, param - param_prev) * dx
            - param_testFunction * ds**2 * dx
        )

        return F
