# C++ with Torch coupling using Pybind11
**note*: experimental*

Pybind11 can be used in 2 variations:
1. Calling C++ code from Python code (in folder `calling`) 
2. Embedding Python code to C++ code (in folder `embedding`)
requires Pybind11 headers

### coupling
`main.cpp` is a dummy CPP program with MPI using Cartesian coordinates that mimic a CFD code\
`calc.py` is the corresponding Torch implementation\
A 3D MPI subdomain in CPP is first transferred to Python\
Python gathers the subdomains and distributes the gathered domain equally to each GPU\
each GPU performs a front propagation and returns the field back to CPP\
use `compile.sh` script to compile and `mysubfile.sh` to submit

### contact:
EI
