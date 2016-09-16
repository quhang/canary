##
# Lassen application.

project(lassen CXX)

# Initializes internal cmake utilities.
include(internal_utils)
config_compiler_and_linker()

# Adds header file search directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# An application is built as a shared library.
cxx_shared_library(lassen "${cxx_default}"
canary-lassen.cc
input.cc
lassen.cc
lassen-util.cc
simulation.cc
simulation-mpi.cc
simulation-parallel.cc
source.cc)

# canary_core library must be built before.
add_dependencies(lassen canary_core)
