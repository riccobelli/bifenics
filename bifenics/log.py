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


from mpi4py import MPI
import time


def log(msg, warning=False, success=False):
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
        print(fmt % (timestamp + msg), flush=True)
