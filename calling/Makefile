# print something
$(info pybind example)

# Define the compiler to use
CC = mpicxx

# Define the compiler flags
CFLAGS = -shared -fPIC -O3 -Wall -std=c++11 $(shell python3 -m pybind11 --includes)

# Define the source files
SRCS = mpi_example.cpp

# Define the executable name
EXE = mpi_example$(shell python3-config --extension-suffix)

# Define the rule to build the executable from the source files
compile: $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(EXE)

# Define the rule to clean the generated files
clean:
	@rm -f $(EXE)

all: clean compile