# C++ with Torch coupling using Pybind11
**note*: experimental*

Pybind11 can be used in 2 variations:
1. Calling C++ code from Python code (in folder `calling`)
2. Embedding Python code to C++ code (in main folder)
requires Pybind11 headers and an MPI implementation --  run `createEnv_pybind.sh` if not

### embedding
`main.cpp` is a dummy C++ program with MPI using Cartesian coordinates that mimic a CFD code\
`calc.py` is the corresponding Torch implementation\
`compile.sh` is to compile C++\

##### notes:
A 3D MPI subdomain in C++ is first transferred to Python\
Python gathers the subdomains and distributes the gathered domain equally to each GPU\
each GPU performs a front propagation for inference and returns the field back to C++\
use `compile.sh` script to compile and `mpirun -n <ranks> example` to run

### contact:
EI