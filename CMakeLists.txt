cmake_minimum_required(VERSION 3.4)
project(example)
set(CMAKE_CXX_STANDARD 11)

find_package(pybind11 REQUIRED) # or `add_subdirectory(pybind11)`
set(pybind11_DIR /p/scratch/raise-ctp2/inanc2/envPybind/lib/python3.10/site-packages/pybind11)

find_package(MPI REQUIRED)

add_executable(example main.cpp)
target_link_libraries(example PRIVATE pybind11::embed MPI::MPI_CXX)