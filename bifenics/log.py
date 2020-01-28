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
        print(fmt % (timestamp + msg))
