# Module collecting several class, each being a general continuation method
from dolfin import (
    Function,
    FiniteElement,
    FunctionSpace,
    MixedElement,
    derivative,
    assign,
    TestFunction,
    TrialFunction,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    split,
    Constant,
    XDMFFile,
    set_log_level,
)
from bifenics.log import log
import os
from mpi4py import MPI
import copy


class ArclengthContinuation(object):
    # Legenda:
    # z: funzione incognita del problema originale
    # param: parametro di controllo del problema di biforcazione
    # ac_state: funzione contente z e param, cioè [z, param]
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
        max_steps=300,
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

    def load_arclength_function(self, func, param, ac_function):
        assign(ac_function.sub(0), func)
        r = Function(self.param_space)
        r.assign(param)
        assign(ac_function.split()[1], r)

    def secant_predictor(
        self, ac_state_prev, ac_state, ds, missing_previous_step=False, omega=1
    ):
        if missing_previous_step is True:
            assign(ac_state_prev, ac_state)
            self.load_arclength_function(
                ac_state.split()[0],
                Constant(ac_state.split()[1] + self._initial_direction * ds),
                ac_state,
            )
            return
        ac_state_bu = ac_state.copy(deepcopy=True)
        predictor = Function(ac_state.function_space())
        predictor.vector()[:] = ac_state.vector()[:] + omega * (
            ac_state.vector()[:] - ac_state_prev.vector()[:]
        )
        assign(ac_state, predictor)
        assign(ac_state_prev, ac_state_bu)

    def save_function(self, function, param, count, xdmf_file):
        # We save the function into the output file
        xdmf_file.write(function, count)
        xdmf_file.write(param, count)

    def run(self):
        # Setting mesh and defining functional spaces
        mesh = self.problem.mesh()
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

        # FIXME: the usage of both a parameter continuation and an arclength
        # continuation uses a HUGE amount of memory! Should be fixed.
        # In the meanwhile use a larger computer.
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
        ac_problem = NonlinearVariationalProblem(residual, ac_state, bcs, J)
        ac_solver = NonlinearVariationalSolver(ac_problem)
        ac_solver.parameters.update(self._solver_params)
        # Start analysis
        count = 0
        n_halving = 0
        while count < self._max_steps and n_halving < self._max_halving:
            log("Computing the predictor (secant method)")
            self.secant_predictor(
                ac_state_prev, ac_state, self._ds, missing_previous_step=missing_prev
            )

            log("Success, starting correction")
            status = ac_solver.solve()
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
                u_copy, param_copy = ac_state.split(deepcopy=True)
                ac_state_copy.assign(ac_state)
                ac_state_prev_copy.assign(ac_state_prev)
                if missing_prev is True:
                    missing_prev = False
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
                # If we have already halved the step five times, we give up.
                if n_halving >= self._max_halving:
                    log("Max halving reached! Ending simulation", warning=True)
