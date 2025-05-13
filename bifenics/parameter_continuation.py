# Copyright (C) 2025 Davide Riccobelli
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

# Module collecting several class, each being a general continuation method
from dolfin import (
    Function,
    derivative,
    TestFunction,
    TrialFunction,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    XDMFFile,
    set_log_level,
)
from bifenics.log import log
import os
from mpi4py import MPI


class ParameterContinuation(object):
    def __init__(
        self,
        problem,
        param_name,
        start=0,
        end=0,
        dt=0,
        min_dt=1e-6,
        save_output=True,
        saving_file_parameters={},
        output_folder="output",
        remove_old_output_folder=True,
        max_dt=1,
        max_step_for_dt_doubling=0,
    ):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        set_log_level(30)

        if (
            rank == 0
            and remove_old_output_folder is True
            and os.path.exists(output_folder)
        ):
            os.system("rm -r " + output_folder)
        if rank == 0 and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.problem = problem
        self._param_name = param_name
        self._param_start = start
        self._param_end = end
        self._dt = dt
        self._min_dt = min_dt
        self._max_dt = max_dt
        self._max_step_for_dt_doubling = max_step_for_dt_doubling
        self._solver_params = {}
        self._save_file = XDMFFile(output_folder + "/results.xdmf")
        self._save_file.parameters.update(saving_file_parameters)
        self._save_output = save_output

        # Update adding user defined solver Parameters
        self._solver_params.update(problem.solver_parameters())

        # Disable error on non nonconvergence
        if "nonlinear_solver" not in self._solver_params:
            self._solver_params["nonlinear_solver"] = "snes"
            self._solver_params["snes_solver"] = {}
        solver_type = self._solver_params["nonlinear_solver"]
        if solver_type == "snes":
            self._solver_params[solver_type + "_solver"][
                "error_on_nonconvergence"
            ] = False

    def run(self):
        # Setting mesh and defining functional spaces
        mesh = self.problem.mesh()
        V = self.problem.function_space(mesh)
        u = Function(V)
        u0 = Function(V)

        # Set initial guess
        u.assign(self.problem.initial_guess(V))
        u0.assign(self.problem.initial_guess(V))

        # Setting parameters values
        self.parameters = self.problem.parameters()
        self.parameters[self._param_name].assign(self._param_start)

        bcs = self.problem.boundary_conditions(mesh, V)
        residual = self.problem.residual(u, TestFunction(V), self.parameters)
        J = derivative(residual, u, TrialFunction(V))

        # Start analysis
        T = 1.0  # total simulation time
        t = 0.0
        log("Parameter continuation started")
        goOn = True
        while round(t, 10) < T and self._dt > 1e-6 and goOn is True:

            t += self._dt
            round(t, 8)
            new_param_value = (
                self._param_start + (self._param_end - self._param_start) * t
            )
            self.parameters[self._param_name].assign(new_param_value)

            log(
                "Percentage completed: "
                + str(round(t * 100, 10))
                + "%"
                + " "
                + self._param_name
                + ": "
                + str(round(new_param_value, 10))
            )

            ok = 0
            n_halving = 0
            if self._solver_params["nonlinear_solver"] == "snes":
                while ok == 0:
                    self.problem.modify_initial_guess(u, self.parameters)
                    status = self.pc_nonlinear_solver(residual, u, bcs, J)
                    if status[1] is True:

                        self.problem.monitor(u, self.parameters, self._save_file)

                        # New solution found, we save it

                        log("Nonlinear solver converged", success=True)
                        if self._save_output is True:
                            self.save_function(
                                u, self.parameters[self._param_name], self._save_file
                            )
                        u0.assign(u)
                        ok = 1
                        if (
                            status[0] <= self._max_step_for_dt_doubling
                            and self._dt < self._max_dt
                        ):
                            log(
                                f"Nonlinear solver converged in less than {self._max_step_for_dt_doubling} "
                                + "iterations, doubling increment of the parameter",
                                success=True,
                            )
                            self._dt = self._dt * 2
                            n_halving += -1
                    else:
                        n_halving += 1
                        # The nonlinear solver failed to converge, we halve the step and
                        # we start again the nonlinear solver.
                        log(
                            "Nonlinear solver did not converge, halving step",
                            warning=True,
                        )
                        self._dt = self._dt / 2.0
                        t += -self._dt
                        new_param_value = (
                            self._param_start
                            + (self._param_end - self._param_start) * t
                        )
                        self.parameters[self._param_name].assign(new_param_value)
                        u.assign(u0)
                        if n_halving > 5:
                            ok = 1
                            log("Max halving reached! Ending simulation", warning=True)
                            goOn = False
            else:
                while ok == 0:
                    self.problem.modify_initial_guess(u, self.parameters)
                    try:
                        self.pc_nonlinear_solver(residual, u, bcs, J)
                        self.problem.monitor(u, self.parameters, self._save_file)

                        # New solution found, we save it

                        log("Nonlinear solver converged", success=True)
                        if self._save_output is True:
                            self.save_function(u, self.parameters, self._save_file)
                        u0.assign(u)
                        ok = 1
                    except RuntimeError:
                        n_halving += 1
                        # The nonlinear solver failed to converge, we halve the step and
                        # we start again the nonlinear solver.
                        log(
                            "Nonlinear solver did not converge, halving step",
                            warning=True,
                        )
                        self._dt = self._dt / 2.0
                        t += -self._dt
                        new_param_value = (
                            self._param_start
                            + (self._param_end - self._param_start) * t
                        )
                        self.parameters[self._param_name].assign(new_param_value)
                        u.assign(u0)
                        if n_halving > 5:
                            ok = 1
                            log("Max halving reached! Ending simulation", warning=True)
                            goOn = False
        return (u, self.parameters)

    def pc_nonlinear_solver(self, residual, u, bcs, J):
        dolfin_problem = NonlinearVariationalProblem(residual, u, bcs, J)
        solver = NonlinearVariationalSolver(dolfin_problem)
        solver.parameters.update(self._solver_params)
        return solver.solve()

    def save_function(self, function, param, xdmf_file):
        # Convert param (which is a Dolfin Constant) to a float, then we take its
        # absolute value
        t = float(param)
        t = round(abs(t), 10)

        # We save the function into the output file
        xdmf_file.write(function, t)
