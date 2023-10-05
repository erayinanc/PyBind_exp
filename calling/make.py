# This is a Python file that imports mpi_example from cpp

# Import mpi4py to initialize and finalize MPI
from mpi4py import MPI

# Import numpy to create an array
import numpy as np

# Import mpi_example to use mpi_sum
import mpi_example

# Torch
import torch

# Create a numpy array of numbers to sum
numbers = np.array([0.0, 0.0, 0.0, 0.0])

# Call mpi_sum with the array and print the result
res = mpi_example.mpi_sum(numbers)
print('sum:',res)

# Generate and print the 3D array using the cpp functions
arr = mpi_example.generate3DArray()
print('array:',arr)

# test cuda for multiGPU device!
for devicen in torch.cuda.device_count():
    torch.cuda.set_device(devicen)
    t1 = torch.zeros(5,5).fill_(1.5).cuda()
    print(f'rank: {torch.cuda.device_count()}, field: {t1}')

# Finalize MPI
MPI.Finalize()