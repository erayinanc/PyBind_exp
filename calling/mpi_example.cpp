// This is a C++ file that defines a function to compute the sum of an array using MPI
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mpi.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cstdio>

using namespace std;
namespace py = pybind11;

// A function that computes the sum of an array using MPI
double mpi_sum(py::array_t<double> array) {
  // Get the rank and size of the MPI communicator
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get the pointer and size of the array
  double* data = (double*) array.request().ptr;
  int n = array.size();

  // Compute the local sum of a part of the array
  int chunk = n / size; // The size of each chunk
  int start = rank * chunk; // The start index of the chunk
  int end = start + chunk; // The end index of the chunk
  if (rank == size - 1) {
    end = n; // The last process takes the remaining elements
  }
  double local_sum = 0.0;
  for (int i = start; i < end; i++) {
    local_sum += data[i] + rank;
  }

  printf("%d\n", rank);

  // Reduce the local sums to get the global sum
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Return the global sum
  return global_sum;
}


// example CFD
// A function to generate a 3D array of size m x n x p
std::vector<std::vector<std::vector<int>>> generate3DArray() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int m = 10;
  int n = 9;
  int p = 11;

  // Create a 3D vector
  std::vector<std::vector<std::vector<int>>> arr(m, std::vector<std::vector<int>>(n, std::vector<int>(p)));

  // Fill the 3D vector with some values
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        arr[i][j][k] = i + j + k;
      }
    }
  }

  // Return the 3D vector
  return arr;
}

// A function to bind the cpp functions to Python
PYBIND11_MODULE(mpi_example, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring
  m.def("generate3DArray", &generate3DArray, "A function to pass a 3D array of size m x n x p");
}