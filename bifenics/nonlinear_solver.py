# Copyright (C) 2024 Bianca Giovanardi
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
#


from dolfin import (
    assemble_system,
    as_backend_type,
    Function,
)
from mpi4py import MPI
from petsc4py import PETSc


class SNESNonlinearSolver(object):
    """
    Generates a new object describing a nonlinear problem
    """

    def __init__(
        self,
        residual, state, bcs, J,):

        # take note of the residual, state, bcs, and Jacobian
        self.residual = residual
        self.state = state

        self.bcs = bcs
        self.J = J

        # initialize the F and J attributes
        self.F_vec = None
        self.J_mat = None
        self.u = None

        # create a new snes object
        self.snes = PETSc.SNES().create(MPI.COMM_WORLD)

        # configure snes
        self.snes.setType("newtonls")
        self.snes.setTolerances(rtol=1e-10, atol=1e-10, max_it=30) # TOFIX: make this a parameter
        self.snes.setMonitor(lambda snes, its, res: print(f"snes iter: {its}, error: {res}"))
        self.snes.setFromOptions()

        # configure ksp
        ksp = self.snes.getKSP()
        # ksp.setMonitor(lambda ksp, its, res: print(f"ksp iter: {its}, error: {res}"))
        ksp.setType('preonly')
        ksp.getPC().setType('lu')
        ksp.getPC().setFactorSolverType('mumps')

    def set_parameters(self, parameters):
        """
        Set parameters for the nonlinear solver
        """

        # show the parameters
        # for key, value in parameters.items():
        #     print(f"{key}: {value}")

        # TOFIX: the parameters are recorded but not applied for now. Set the paramters as options
        #        in the petsc solver

        # take note of the parameters
        self.parameters = parameters

    def set_null_space(self, null_space):
        # get the snes KSP object
        ksp = self.snes.getKSP()
        # get the matrix
        [A, _] = ksp.getOperators()
        # set the null space
        A.setNullSpace(null_space)
        # all done
        pass

    def F_func(self, snes, x, F):
        # update the state with the current guess
        x.copy(self.state.vector().vec())

        # assemble the system
        J_mat, F_vec = assemble_system(self.J, self.residual, self.bcs)

        # assemble the matrix
        self.J_mat = as_backend_type(J_mat).mat()
        self.J_mat.assemblyBegin()
        self.J_mat.assemblyEnd()

        # assemble the vector
        self.F_vec = as_backend_type(F_vec).vec()
        self.F_vec.assemblyBegin()
        self.F_vec.assemblyEnd()

        # print the norms of the matrix and the vector
        # print(PETSc.Mat.norm(self.J_mat))
        # print(PETSc.Vec.norm(self.F_vec))
        # print(f"snes iter: {snes.getIterationNumber()}, error: {PETSc.Vec.norm(self.F_vec)}")

        F.copy(self.F_vec)

        pass

    def J_func(self, snes, x, J, P):
        J.copy(self.J_mat)

        pass

    def solve(self):
        # assemble the system to get the size of F_mat and J_func
        J_mat, F_vec = assemble_system(self.J, self.residual, self.bcs)

        # set the function and Jacobian
        self.snes.setFunction(self.F_func, as_backend_type(F_vec).vec())
        self.snes.setJacobian(self.J_func, as_backend_type(J_mat).mat())

        # TOFIX: this would be the proper way to do it
        # b = self.init_residual()
        # A = self.init_jacobian()
        # snes.setFunction(self.F_func, b.vec())
        # snes.setJacobian(self.J_func, A.mat())

        # make a copy of the state to use as initial guess
        self.u = as_backend_type(self.state.copy(deepcopy=True).vector()).vec()

        # solve the system
        self.snes.solve(None, self.u)

        # print the reason for convergence
        # print(self.snes.getConvergedReason())

        # return the number of iterations and whether the solver converged
        return [self.snes.its, self.snes.is_converged]
