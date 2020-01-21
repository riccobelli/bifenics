# Module collecting several class, each being a general continuation method
from dolfin import Function, derivative, TestFunction, TrialFunction, \
    NonlinearVariationalProblem, NonlinearVariationalSolver
from mpi4py import MPI
import time


class ParameterContinuation(object):

    def __init__(self, problem, param_name, start=0, end=0, dt=0, min_dt=1e-6):
        self.problem = problem
        self._param_name = param_name
        self._param_start = start
        self._param_end = end
        self._dt = dt
        self._min_dt = min_dt
        self._solver_params = {'nonlinear_solver': 'snes'}

        # Update adding user defined solver Parameters
        self._solver_params.update(problem.solver_parameters())
        # Disable error on SNES non nonconvergence
        if 'nonlinear_solver' not in self._solver_params:
            self._solver_params['nonlinear_solver'] == 'snes'
        solver_type = self._solver_params['nonlinear_solver']
        self._solver_params[solver_type +
                            '_solver']['error_on_nonconvergence'] = False

    def log(self, msg, warning=False, success=False):
        # Function for printing log messages in parallel

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0 and warning:
            fmt = "\033[1;37;31m%s\033[0m"  # Red
        elif rank == 0 and success:
            fmt = "\033[1;37;32m%s\033[0m"  # Green
        elif rank == 0:
            fmt = "\033[1;37;34m%s\033[0m"  # Blue
        if rank == 0:
            timestamp = "[%s] " % time.strftime("%H:%M:%S")
            print(fmt % (timestamp + msg))

    def pc_nonlinear_solver(self, residual, u, bcs, J):
        dolfin_problem = NonlinearVariationalProblem(residual, u, bcs, J)
        solver = NonlinearVariationalSolver(dolfin_problem)
        solver.parameters.update(self._solver_params)
        return solver.solve()

    def run(self):
        # Setting mesh and defining functional spaces
        mesh = self.problem.mesh()
        V = self.problem.function_space(mesh)
        u = Function(V)
        u0 = Function(V)

        # Setting parameters values
        self.param = self.problem.parameters()[self._param_name]
        self.param.assign(self._param_start)

        bcs = self.problem.boundary_conditions(mesh, V)
        residual = self.problem.residual(u, TestFunction(V), self.param)
        J = derivative(residual, u, TrialFunction(V))

        # Start analysis
        T = 1.0  # total simulation time
        t = 0.0
        self.log("Parameter continuation started")
        while round(t, 10) < T and self._dt > 1e-6:

            t += self._dt
            round(t, 8)
            self.param.assign(
                self._param_start + (self._param_end - self._param_start) * t)

            self.log(
                "Percentage completed: " + str(round(t * 100, 10)) + "%" + " "
                + self._param_name + ": " + str(round(float(self.param), 10)))

            ok = 0
            while ok == 0:
                status = self.pc_nonlinear_solver(residual, u, bcs, J)
                if status[1] is True:
                    # New solution found, we save it
                    self.problem.monitor()
                    self.log("Nonlinear solver converged", success=True)
                    u0.assign(u)
                    ok = 1
                else:
                    # The nonlinear solver failed to converge, we halve the
                    # step and we start again the nonlinear solver.
                    self.log(
                        "Nonlinear solver did not converge, halving step",
                        warning=False)
                    self._dt = self._dt / 2.
                    t += -self._dt
                    self.param.assign(
                        self._param_start +
                        (self._param_end - self._param_start) * t)
                    print(float(self.param))
                    u.assign(u0)
