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
    Expression,
    split,
    interpolate,
    Constant,
    plot,
    project,
    XDMFFile)
from bifenics.log import log
import os
from mpi4py import MPI
import matplotlib.pyplot as plt


class ParameterContinuation(object):

    def __init__(self,
                 problem,
                 param_name,
                 start=0,
                 end=0,
                 dt=0,
                 min_dt=1e-6,
                 save_output=True,
                 saving_file_parameters={},
                 output_folder="output",
                 remove_old_output_folder=True):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0 and remove_old_output_folder is True:
            os.system("rm -r " + output_folder)
        if rank == 0 and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.problem = problem
        self._param_name = param_name
        self._param_start = start
        self._param_end = end
        self._dt = dt
        self._min_dt = min_dt
        self._solver_params = {}
        self._save_file = XDMFFile(output_folder + "/results.xdmf")
        self._save_file.parameters.update(saving_file_parameters)
        self._save_output = save_output

        # Update adding user defined solver Parameters
        self._solver_params.update(problem.solver_parameters())

        # Disable error on non nonconvergence
        if 'nonlinear_solver' not in self._solver_params:
            self._solver_params['nonlinear_solver'] = 'snes'
            self._solver_params['snes_solver'] = {}
        solver_type = self._solver_params['nonlinear_solver']
        if solver_type == "snes":
            self._solver_params[solver_type + '_solver']['error_on_nonconvergence'] = False

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
        param = self.problem.parameters()[self._param_name]
        param.assign(self._param_start)

        bcs = self.problem.boundary_conditions(mesh, V)
        residual = self.problem.residual(u, TestFunction(V), param)
        J = derivative(residual, u, TrialFunction(V))

        # Start analysis
        T = 1.0  # total simulation time
        t = 0.0
        log("Parameter continuation started")
        goOn = True
        while round(t, 10) < T and self._dt > 1e-6 and goOn is True:

            t += self._dt
            round(t, 8)
            param.assign(self._param_start + (self._param_end - self._param_start) * t)

            log("Percentage completed: " + str(round(t * 100, 10)) + "%" +
                " " + self._param_name + ": " + str(round(float(param), 10)))

            ok = 0
            n_halving = 0
            if self._solver_params['nonlinear_solver'] == 'snes':
                while ok == 0:
                    self.problem.modify_initial_guess(u, param)
                    status = self.pc_nonlinear_solver(residual, u, bcs, J)
                    if status[1] is True:

                        self.problem.monitor(u, param, self._save_file)

                        # New solution found, we save it

                        log("Nonlinear solver converged", success=True)
                        if self._save_output is True:
                            self.save_function(u, param, self._save_file)
                        u0.assign(u)
                        ok = 1
                    else:
                        n_halving += 1
                        # The nonlinear solver failed to converge, we halve the step and we start
                        # again the nonlinear solver.
                        log("Nonlinear solver did not converge, halving step", warning=True)
                        self._dt = self._dt / 2.
                        t += -self._dt
                        param.assign(self._param_start + (self._param_end - self._param_start) * t)
                        u.assign(u0)
                        if n_halving > 5:
                            ok = 1
                            log("Max halving reached! Ending simulation", warning=True)
                            goOn = False
            else:
                raise NotImplementedError
                while ok == 0:
                    self.problem.modify_initial_guess(u, param)
                    try:
                        self.pc_nonlinear_solver(residual, u, bcs, J)
                        self.problem.monitor(u, param, self._save_file)

                        # New solution found, we save it

                        log("Nonlinear solver converged", success=True)
                        if self._save_output is True:
                            self.save_function(u, param, self._save_file)
                        u0.assign(u)
                        ok = 1
                    except BaseException:
                        n_halving += 1
                        # The nonlinear solver failed to converge, we halve the step and we
                        # start again the nonlinear solver.
                        log("Nonlinear solver did not converge, halving step", warning=True)
                        self._dt = self._dt / 2.
                        t += -self._dt
                        param.assign(self._param_start +
                                     (self._param_end - self._param_start) * t)
                        u.assign(u0)
                        if n_halving > 5:
                            ok = 1
                            log("Max halving reached! Ending simulation", warning=True)
                            goOn = False

    def pc_nonlinear_solver(self, residual, u, bcs, J):
        dolfin_problem = NonlinearVariationalProblem(residual, u, bcs, J)
        solver = NonlinearVariationalSolver(dolfin_problem)
        solver.parameters.update(self._solver_params)
        return solver.solve()

    def save_function(self, function, param, xdmf_file):
        # Convert param (which is a Dolfin Constant) to a float, then we take its absolute value
        t = float(param)
        t = round(abs(t), 10)

        # We save the function into the output file
        xdmf_file.write(function, t)


class ArclengthContinuation(object):
    # Legenda:
    # z: funzione incognita del problema originale
    # param: parametro di controllo del problema di biforcazione
    # ac_state: funzione contente z e param, cioè [z, param]
    def __init__(self,
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
                 max_steps=100):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0 and remove_old_output_folder is True:
            os.system("rm -r " + output_folder)
        if rank == 0 and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.problem = problem
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

        # Update adding user defined solver Parameters
        self._solver_params.update(problem.solver_parameters())

        # Disable error on non nonconvergence
        if 'nonlinear_solver' not in self._solver_params:
            self._solver_params['nonlinear_solver'] = 'snes'
            self._solver_params['snes_solver'] = {}
        solver_type = self._solver_params['nonlinear_solver']
        if solver_type == "snes":
            self._solver_params[solver_type + '_solver']['error_on_nonconvergence'] = False

    def load_arclength_function(self, func, param, ac_function):
        assign(ac_function.sub(0), func)
        r = Function(self.param_space)
        r.assign(param)
        assign(ac_function.split()[1], r)

    def secant_predictor(self, ac_state_prev, ac_state, ds, missing_previous_step=False, omega=1):
        if missing_previous_step is True:
            assign(ac_state_prev, ac_state)
            self.load_arclength_function(
                ac_state.split()[0],
                Constant(ac_state.split()[1] + ds),
                ac_state)
            return
        ac_state_bu = ac_state.copy(deepcopy=True)
        predictor = project(ac_state + omega * (ac_state - ac_state_prev),
                            ac_state.function_space())
        assign(ac_state, predictor)
        assign(ac_state_prev, ac_state_bu)

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

        # Setting parameters values
        param = self.problem.parameters()[self._param_name]
        param.assign(self._param_start)

        # Loading initial arclength state
        initial_guess = self.problem.initial_guess(V_space)
        self.load_arclength_function(initial_guess, param, ac_state)
        self.load_arclength_function(initial_guess, param, ac_state_prev)

        # Boundary conditions
        bcs = self.problem.boundary_conditions(mesh, ac_space.sub(0))

        # Construction of the arclength problem residual and Jacobian starting from the
        # user defined ones.
        u, param = split(ac_state)
        u_prev, param_prev = split(ac_state_prev)

        ac_testFunction = TestFunction(ac_space)
        u_testFunction, param_testFunction = split(ac_testFunction)
        residual = (self.problem.residual(u, u_testFunction, param) +
                    self.problem.ac_constraint(ac_state, ac_state_prev, ac_testFunction, self._ds))
        J = derivative(residual, ac_state, TrialFunction(ac_space))
        ac_problem = NonlinearVariationalProblem(residual, ac_state, bcs, J)
        ac_solver = NonlinearVariationalSolver(ac_problem)
        # Start analysis
        count = 0
        self.secant_predictor(ac_state_prev, ac_state, self._ds, missing_previous_step=True)
        while count < self._max_steps:
            self.secant_predictor(ac_state_prev, ac_state, self._ds)
            J = derivative(residual, ac_state, TrialFunction(ac_space))
            ac_problem = NonlinearVariationalProblem(residual, ac_state, bcs, J)
            ac_solver = NonlinearVariationalSolver(ac_problem)
            ac_solver.solve()
            u_copy, param_copy = ac_state.split(deepcopy=True)
            print(float(param_copy))
            self.problem.monitor(u_copy, float(param_copy), self._save_file)
            count += 1
