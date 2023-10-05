// This is a dummy CPP program with MPI using Caartesian coordinates that mimic a CFD code
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <mpi.h>
#include <vector>
#include <iostream>

namespace py = pybind11;
using namespace std;

int main(int argc, char **argv) {
    // MPI
    // parameters
    int rank, size;
    int dims[3] = {0, 0, 0}; // let MPI decide the dimensions
    int periods[3] = {1, 1, 1}; // enable wraparound
    int coords[3];
    int up_rank, down_rank, left_rank, right_rank, top_rank, bottom_rank;
    //int msg = rank; // the message is the rank of the sender
    //int recv_msg; // the received message

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create a 3D Cartesian topology with wraparound
    MPI_Comm cart_comm; // the new communicator
    MPI_Dims_create(size, 3, dims); // compute the dimensions
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm); // create the topology

    // Get the coordinates of the current process in the grid
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    //cout << "MPI_Cart_info: rank:" << rank << " / coords:" << coords[0] << "," << coords[1] << "," << coords[2]
    //          << " / dims: " << dims[0] << "," << dims[1] << "," << dims[2] << endl;
    if (rank == 0) {
        cout << "MPI:" << endl;
        cout << "MPI_Cart_info: size:" << size << endl;
    }

    // Get the ranks of the neighboring processes
    MPI_Cart_shift(cart_comm, 0, 1, &left_rank, &right_rank); // shift along the 1st axis
    MPI_Cart_shift(cart_comm, 1, 1, &up_rank, &down_rank); // shift along the 2nd axis
    MPI_Cart_shift(cart_comm, 2, 1, &top_rank, &bottom_rank); // shift along the 3rd axis

    /* example to sendrecv data -- for test purposes -- not needed
    // Send and receive messages along the edges
    MPI_Status status; // the status of the communication

    // Send to the right and receive from the left
    MPI_Sendrecv(&msg, 1, MPI_INT, right_rank, 0,
                 &recv_msg, 1, MPI_INT, left_rank, 0,
                 cart_comm, &status);
    cout << "Process " << rank << " at (" << coords[0] << ", " <<
        coords[1] << ") received " << recv_msg << " from the left" << endl;

    // reobtain Cartesian info
    //MPI_Cart_get(cart_comm, 2, dims, periods, coords);
    */

    // PyBind
    // parameters
    const int n = 14; // Array dims to be passed to Torch
    const int m = 14; //
    const int l = 1; //
    vector<int> d_glob(3);
    vector<int> n_glob(2);
    vector<int> m_glob(2);
    vector<int> l_glob(2);

    // variables
    // get global array dims
    d_glob[0] = n * dims[0];
    d_glob[1] = m * dims[1];
    d_glob[2] = l * dims[2];
    // get global coordinates
    n_glob[0] = d_glob[0] * coords[0];
    n_glob[1] = d_glob[0] * (coords[0] + 1);
    m_glob[0] = d_glob[1] * coords[0];
    m_glob[1] = d_glob[1] * (coords[0] + 1);
    l_glob[0] = d_glob[2] * coords[0];
    l_glob[1] = d_glob[2] * (coords[0] + 1);

    // declare a dummy 3D array with (n,m,l) dimensions to be transferred to torch
    vector<vector<vector<double>>> dummy_array(n, vector<vector<double>>(m, vector<double>(l)));

    // fill w/ random numbers
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < l; k++) {
                dummy_array[i][j][k] = rand() % 100;
            }
        }
    }

    // Initialize Python interpreter and load a script
    py::scoped_interpreter guard{};
    py::module script = py::module::import("calc");

    // Pass Cartesian topology to Torch
    script.attr("get_topology")(rank,d_glob,n_glob,m_glob,l_glob);

    // Pass a field in an MPI rank
    int res = script.attr("torch_couple")(dummy_array).cast<double>();
    if (rank == 0) {
        cout << "\nCPP:" << endl;
        cout << "res = " << res << endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

//eof