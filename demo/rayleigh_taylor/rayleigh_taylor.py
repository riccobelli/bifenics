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

# We consider a rectangle, composed of a slightly compressible neo-Hookean
# material and subjected to a body force gamma along the y direction. It
# is inspired by the problem described in

# Mora, S., Phou, T., Fromental, J. M., & Pomeau, Y. (2014). "Gravity
# driven instability in elastic solid layers". Physical Review Letters,
# 113(17), 178301.

from bifenics import BifenicsProblem, ParameterContinuation
from dolfin import (
    RectangleMesh,
    Point,
    VectorFunctionSpace,
    grad,
    Identity,
    inner,
    derivative,
    dx,
    tr,
    Constant,
    ln,
    det,
    SubDomain,
    near,
    DirichletBC,
    MeshFunction,
    sin,
    cos,
    DOLFIN_PI,
)
import numpy as np


class RayleighTaylor(BifenicsProblem):
    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            TOL = 1e-4
            return on_boundary and near(x[1], 0.0, TOL)

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            TOL = 1e-4
            return on_boundary and near(x[0], 0.0, TOL)

    class Right(SubDomain):
        def __init__(self, L):
            self.L = L
            SubDomain.__init__(self)  # Call base class constructor!

        def inside(self, x, on_boundary):
            TOL = 1e-4
            return on_boundary and near(x[0], self.L, TOL)

    def __init__(self, L, H, mu, K, nx=30, ny=30):
        self.L = L
        self.H = H
        self.nx = nx
        self.ny = ny

        # Elastic constants
        self.mu = Constant(mu)
        self.K = Constant(K)

    def mesh(self):
        mesh = RectangleMesh(
            Point((0, 0)), Point((self.L, self.H)), self.nx, self.ny, "left/right"
        )
        x = mesh.coordinates()[:, 0]
        y = mesh.coordinates()[:, 1]

        m = 1
        for elem in range(len(x)):
            if y[elem] > 0.98 * self.H:
                x[elem] = x[elem] + 0.001 * sin(2 * m * DOLFIN_PI / self.L * x[elem])
                y[elem] = y[elem] + 0.001 * cos(2 * m * DOLFIN_PI / self.L * x[elem])
        xy_coor = np.array([x, y]).transpose()
        mesh.coordinates()[:] = xy_coor
        return mesh

    def function_space(self, mesh):
        return VectorFunctionSpace(mesh, "CG", 1)

    def parameters(self):
        return {"gamma": Constant(0)}

    def boundary_conditions(self, mesh, V):
        bottom = self.Bottom()
        left = self.Left()
        right = self.Right(self.L)

        boundaries = MeshFunction("size_t", mesh, 1)
        boundaries.set_all(0)
        bottom.mark(boundaries, 1)
        left.mark(boundaries, 2)
        right.mark(boundaries, 3)

        bcb = DirichletBC(V, Constant((0, 0)), boundaries, 1)
        bcl = DirichletBC(V.sub(0), Constant(0), boundaries, 2)
        bcr = DirichletBC(V.sub(0), Constant(0), boundaries, 3)

        return [bcb, bcl, bcr]

    def residual(self, u, v, parameters):
        gamma = parameters["gamma"]
        F = Identity(2) + grad(u)
        C = F.T * F
        J = det(F)
        W = self.mu * (tr(C) - 2 * ln(J)) + self.K * ln(J) * ln(J)

        ext_forces = gamma * inner(u, Constant((0, 1)))
        psi = (W - ext_forces) * dx

        return derivative(psi, u, v)

    def solver_parameters(self):
        parameters = {
            "nonlinear_solver": "snes",
            "snes_solver": {
                "linear_solver": "mumps",
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 30,
            },
        }
        return parameters


if __name__ == "__main__":
    XDMF_options = {
        "flush_output": True,
        "functions_share_mesh": True,
        "rewrite_function_mesh": False,
    }
    rt = RayleighTaylor(2, 1, 1, 200, nx=60, ny=30)
    analysis = ParameterContinuation(
        rt, "gamma", start=0, end=15, dt=0.01, saving_file_parameters=XDMF_options
    )
    analysis.run()
