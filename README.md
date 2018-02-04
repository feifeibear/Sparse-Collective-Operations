# `Sparse Allreduce`
Sparse allreduce based on MPI

## Installation
```bash
make
#mpirun -np #nodes ./allreduce-test (cpu|gpu) Sparse_Ratio
mpirun -np 4 ./allreduce-test cpu 0.1
```
