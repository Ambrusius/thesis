from mpi4py import MPI
import time

def main(rank, ws):
    time.sleep(1)

def worker(rank, ws):
    time.sleep(1)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ws = comm.Get_size() - 1
print(rank)
if rank == 0:
    time_start = time.time()
    main(rank, ws)
    time_finish = time.time()
    print(f'Time taken: {time_finish - time_start} seconds, using {ws} workers')
else:
    worker(rank, ws)