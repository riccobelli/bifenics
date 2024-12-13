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


# This file implements an arclength continuation algorithm.

from dolfin import (
    Function,
    FiniteElement,
    FunctionSpace,
    FunctionAssigner,
    MixedElement,
    derivative,
    assign,
    TestFunction,
    TestFunctions,
    TrialFunction,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    split,
    inner,
    Constant,
    XDMFFile,
    dx,
    assemble,
    set_log_level,
)
from bifenics.log import log
import os
from mpi4py import MPI
from bifenics.nonlinear_solver import SNESNonlinearSolver
import copy


class ArclengthContinuation(object):
    def __init__(
        self,
        problem,
        param_name,
        start=0,
        end=0,
        ds=0,
        min_ds=1e-6,
        save_output=True,
        saving_file_parameters={},
        output_folder="output",
        remove_old_output_folder=True,
        initial_direction=1,
        max_halving=10,
        first_step_with_parameter_continuation=False,
        n_step_for_doubling=0,
        max_steps=300,
        predictor_type="tangent",  # tangent or secant
    ):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        set_log_level(40)

        if (
            rank == 0
            and remove_old_output_folder is True
            and os.path.exists(output_folder)
        ):
            os.system("rm -r " + output_folder)
        if rank == 0 and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.problem = problem
        self._max_halving = max_halving
        self._param_name = param_name
        self._param_start = start
        self._param_end = end
        self._ds = Constant(ds)
        self._min_ds = min_ds
        self._solver_params = {}
        self._save_file = XDMFFile(output_folder + "/results.xdmf")
        self._save_file.parameters.update(saving_file_parameters)
        self._save_output = save_output
        self._max_steps = max_steps
        self._initial_direction = initial_direction
        self._first_step_with_parameter_continuation = (
            first_step_with_parameter_continuation
        )
        self.n_step_for_doubling = n_step_for_doubling

        # Update adding user defined solver Parameters
        self._solver_params.update(problem.solver_parameters())

        self.predictor_type = predictor_type

        # Disable error on non nonconvergence
        if "nonlinear_solver" not in self._solver_params:
            self._solver_params["nonlinear_solver"] = "snes"
            self._solver_params["snes_solver"] = {}
        solver_type = self._solver_params["nonlinear_solver"]
        if solver_type == "snes":
            self._solver_params[solver_type + "_solver"][
                "error_on_nonconvergence"
            ] = False

    def load_arclength_function(self, func, param, ac_function):
        assign(ac_function.sub(0), func)
        r = Function(self.param_space)
        r.assign(param)
        assign(ac_function.split()[1], r)

    def secant_predictor(
        self, ac_state_prev, ac_state, ds, missing_previous_step=False, omega=1
    ):
        ac_space = ac_state.function_space()
        predictor = Function(ac_space)
        if missing_previous_step is True:
            state, param = ac_state.split(deepcopy=True)
            self.load_arclength_function(
                state,
                Constant(param + ds * self._initial_direction),
                predictor,
            )
        else:
            predictor.vector()[:] = ac_state.vector()[:] + omega * (
                ac_state.vector()[:] - ac_state_prev.vector()[:]
            )
        return predictor

    def tangent_predictor(
        self, old_predictor, ac_state, ds, bcs, missing_previous_step=False, omega=1
    ):
        ac_state_bu = ac_state.copy(deepcopy=True)
        V = ac_state.function_space()
        predictor = Function(V)
        u, ac_param = split(ac_state)
        predictor_u, predictor_param = split(predictor)
        v, mu = TestFunctions(V)
        state_residual = self.problem.residual(u, v, self.parameters)
        if missing_previous_step is True:
            normalization = mu * (self._initial_direction - predictor_param) * dx
            old_predictor = Function(V)
        else:
            normalization = mu * (inner(predictor, old_predictor) - Constant(1)) * dx
        tangent_residual = (
            derivative(state_residual, ac_state, predictor) + normalization
        )
        tangent_jacoobian = derivative(tangent_residual, predictor, TrialFunction(V))
        tangent_problem = NonlinearVariationalProblem(
            tangent_residual, predictor, bcs, tangent_jacoobian
        )
        tangent_solver = NonlinearVariationalSolver(tangent_problem)
        tangent_solver.parameters.update(self._solver_params)
        status = tangent_solver.solve()

        if status[1] is True:
            old_predictor.assign(predictor)
        return (predictor, status[1])

    def save_function(self, function, param, count, xdmf_file):
        # We save the function into the output file
        xdmf_file.write(function, count)
        xdmf_file.write(param, count)

    def run(self):
        # Setting mesh and defining functional spaces
        mesh = self.problem.mesh()
        # take note of the problem's null space
        self._null_space = self.problem.null_space()
        V_space = self.problem.function_space(mesh)
        V_elem = V_space.ufl_element()
        param_elem = FiniteElement("R", mesh.ufl_cell(), 0)
        self.param_space = FunctionSpace(mesh, param_elem)
        ac_element = MixedElement([V_elem, param_elem])
        ac_space = FunctionSpace(mesh, ac_element)

        # Creating functions in arclength spaces
        ac_state = Function(ac_space)
        ac_state_prev = Function(ac_space)
        ac_state_copy = Function(ac_space)
        ac_state_prev_copy = Function(ac_space)

        # Setting parameters values
        self.parameters = self.problem.parameters()
        self.parameters[self._param_name].assign(self._param_start)

        # Loading initial arclength state
        if self._first_step_with_parameter_continuation is False:
            initial_guess = self.problem.initial_guess(V_space)
            actual_param = self.parameters[self._param_name]
            self.load_arclength_function(initial_guess, actual_param, ac_state)
            self.load_arclength_function(initial_guess, actual_param, ac_state_prev)
            missing_prev = True

        # FIXME: the use of both a parameter continuation and an arclength
        # continuation uses a HUGE amount of memory! Should be fixed.
        else:
            log("Computing first step with a parameter continuation")
            u = Function(V_space)

            # Set initial guess
            u.assign(self.problem.initial_guess(V_space))

            bcs = self.problem.boundary_conditions(mesh, V_space)
            residual = self.problem.residual(u, TestFunction(V_space), self.parameters)
            J = derivative(residual, u, TrialFunction(V_space))
            dolfin_problem = NonlinearVariationalProblem(residual, u, bcs, J)
            solver = NonlinearVariationalSolver(dolfin_problem)
            initial_solver_param = copy.deepcopy(self._solver_params)
            solver_type = initial_solver_param["nonlinear_solver"]
            initial_solver_param[solver_type + "_solver"][
                "error_on_nonconvergence"
            ] = True
            solver.parameters.update(initial_solver_param)
            solver.solve()
            log("Success", success=True)
            # First solution found, we save it on ac_state_prev
            actual_param = self.parameters[self._param_name]
            self.load_arclength_function(u, actual_param, ac_state_prev)

            # We look for the next one
            log("Computing second step with a parameter continuation")
            new_param_value = actual_param + self._initial_direction * self._ds
            self.parameters[self._param_name].assign(new_param_value)
            dolfin_problem = NonlinearVariationalProblem(residual, u, bcs, J)
            solver = NonlinearVariationalSolver(dolfin_problem)
            solver.parameters.update(initial_solver_param)
            solver.solve()

            # We succeded! (We hope, otherwise our adventure ends here). We save
            # the solution
            actual_param = self.parameters[self._param_name]
            self.load_arclength_function(u, actual_param, ac_state)
            log("Success", success=True)
            missing_prev = False
        ac_state_copy.assign(ac_state)
        ac_state_prev_copy.assign(ac_state_prev)
        # Boundary conditions
        bcs = self.problem.boundary_conditions(mesh, ac_space.sub(0))

        # Construction of the arclength problem residual and Jacobian starting from the
        # user defined ones.
        u, ac_param = split(ac_state)
        self.parameters[self._param_name] = ac_param
        u_prev, ac_param_prev = split(ac_state_prev)

        ac_testFunction = TestFunction(ac_space)
        u_testFunction, param_testFunction = split(ac_testFunction)
        residual = self.problem.residual(
            u, u_testFunction, self.parameters
        ) + self.problem.ac_constraint(
            ac_state, ac_state_prev, ac_testFunction, self._ds
        )
        J = derivative(residual, ac_state, TrialFunction(ac_space))
        # ac_problem = NonlinearVariationalProblem(residual, ac_state, bcs, J)

        # by default, say there are no constraints
        constraints_present = False
        try:
            # fetch the primal variable
            u = ac_state.sub(0)
            # check if the problem requires constraints on the primal variable
            [v_lower_bound, v_upper_bound] = self.problem.constraints(u)
            # then there are constraints
            constraints_present = True
        except:
            # nothing to be done
            pass

        # create a SNES solver
        ac_solver = SNESNonlinearSolver(residual, ac_state, bcs, J)

        # if the problem has a null space, attach it to the solver
        if self._null_space:
            ac_solver.set_null_space(self._null_space)

        # Start analysis
        count = 0
        n_halving = 0
        omega = 1  # Correction for the secant predictor in presence of halvings
        old_tangent = Function(ac_space)
        self.load_arclength_function(
            old_tangent.split()[0],
            Constant(ac_state.split()[1] + self._initial_direction),
            old_tangent,
        )
        param_copy = Constant(self._param_start)

        while (
            count < self._max_steps
            and n_halving < self._max_halving
            and min(self._param_start, self._param_end)
            <= float(param_copy)
            <= max(self._param_start, self._param_end)
        ):
            log(f"Computing the predictor ({self.predictor_type} method)")
            ac_state_bu = ac_state.copy(deepcopy=True)
            if self.predictor_type == "secant":
                predictor = self.secant_predictor(
                    ac_state_prev,
                    ac_state,
                    self._ds,
                    missing_previous_step=missing_prev,
                    omega=omega,
                )
            elif self.predictor_type == "tangent":
                tangent, tangent_solver_success = self.tangent_predictor(
                    old_tangent,
                    ac_state,
                    self._ds,
                    bcs,
                    missing_previous_step=missing_prev,
                    omega=omega,
                )
                if tangent_solver_success is False:
                    log(
                        "Tangent solver did not converge! Fallback to secant predictor",
                        warning=True,
                    )
                    predictor = self.secant_predictor(
                        ac_state_prev, ac_state, self._ds, missing_prev, omega
                    )
                else:
                    predictor = Function(ac_space)
                    predictor.vector()[:] = (
                        ac_state.vector() + self._ds * tangent.vector()
                    )
            else:
                raise ValueError("Predictor not implemented. Modify the predictor_type")
            assign(ac_state, predictor)
            assign(ac_state_prev, ac_state_bu)
            log("Success, starting correction")
            status = ac_solver.solve()
            ac_state = ac_solver.state
            if status[1] is True:
                # The nonlinear solver reached convergence, we need to save the solution
                # and to prepare for the next step, first we separate the solution to
                # the original problem and the parameter
                log("Nonlinear solver converged", success=True)
                u_copy, param_copy = ac_state.split(deepcopy=True)
                # We call the monitor to execute tasks on the solution
                log(f"Step {count}: {self._param_name} = {float(param_copy)}")
                self.problem.ac_monitor(
                    u_copy, param_copy, self.parameters, self._save_file
                )
                # If needed, we save the file. At the same time-step (i.e. the value of
                # "count"), we save both the solution of the original problem and the
                # parameter
                if self._save_output is True:
                    self.save_function(u_copy, param_copy, count, self._save_file)
                # Ready for the next step! We increment count and we "back up" our
                # solution
                count += 1
                ac_state_copy.assign(ac_state)
                ac_state_prev_copy.assign(ac_state_prev)
                if missing_prev is True:
                    missing_prev = False
                if status[0] <= self.n_step_for_doubling and n_halving > 0:
                    self._ds.assign(self._ds * 2)
                    omega = omega * 2
                    n_halving += -1
                    log(
                        f"Converged in less than {self.n_step_for_doubling+1} steps, "
                        + "doubling ds",
                        success=True,
                    )
                else:
                    omega = 1
            else:
                # The nonlinear solver failed to converge, we halve the step and we
                # start again the nonlinear solver.
                n_halving += 1
                log("Nonlinear solver did not converge, halving step", warning=True)
                self._ds.assign(self._ds / 2)
                # We restore the previous state, deleting the increment given by the
                # predictor.
                ac_state.assign(ac_state_copy)
                ac_state_prev.assign(ac_state_prev_copy)
                omega = omega / 2
                # If we have already halved the step five times, we give up.
                if n_halving >= self._max_halving:
                    log("Max halving reached! Ending simulation", warning=True)
        return (ac_state, self.parameters)
