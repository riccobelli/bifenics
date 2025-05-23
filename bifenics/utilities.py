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


# This file implements some functions that may be useful in the post-processing
# when the code runs in parallel.


from dolfin import Point, DOLFIN_EPS
import numpy as np


def evaluate_function(u, x):
    mesh = u.function_space().mesh()
    comm = mesh.mpi_comm()
    if comm.size == 1:
        return u(*x)

    # Find whether the point lies on the partition of the mesh local
    # to this process, and evaulate u(x)
    cell, distance = mesh.bounding_box_tree().compute_closest_entity(Point(*x))
    u_eval = u(*x) if distance < DOLFIN_EPS else None

    # Gather the results on process 0
    comm = mesh.mpi_comm()
    computed_u = comm.gather(u_eval, root=0)

    # Verify the results on process 0 to ensure we see the same value
    # on a process boundary
    if comm.rank == 0:
        global_u_evals = np.array(
            [y for y in computed_u if y is not None], dtype=np.double
        )
        assert np.all(np.abs(global_u_evals[0] - global_u_evals) < 1e-9)

        computed_u = global_u_evals[0]
    else:
        computed_u = None

    # Broadcast the verified result to all processes
    computed_u = comm.bcast(computed_u, root=0)

    return computed_u


def parallelizemaxormin(maxormin):
    def wrapper(x, comm):
        if comm.size == 1:
            return maxormin(x, comm)

        # Find the maximum on each process
        if len(x > 0):
            maxormin_proc = maxormin(x, comm)
        else:
            maxormin_proc = None

        # Gather the results on process 0
        computed_maxormin = comm.gather(maxormin_proc, root=0)

        # Verify the results on process 0 to ensure we see the same value
        # on a process boundary
        if comm.rank == 0:
            computed_maxormin = np.array(
                [y for y in computed_maxormin if y is not None], dtype=np.double
            )
            maxormin_proc = maxormin(computed_maxormin, comm)

        # Broadcast the verified result to all processes
        computed_maxormin = comm.bcast(maxormin_proc, root=0)

        return computed_maxormin

    return wrapper


@parallelizemaxormin
def parallel_max(x, comm):
    return np.max(x)


@parallelizemaxormin
def parallel_min(x, comm):
    return np.min(x)
