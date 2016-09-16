##
# Logistic regression application running a fixed number of loops.

project(logistic_loop CXX)

# Initializes internal cmake utilities.
include(internal_utils)
config_compiler_and_linker()

# Adds header file search directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# An application is built as a shared library.
cxx_shared_library(logistic_loop "${cxx_default}" logistic_loop.cc)
# canary_core library must be built before.
add_dependencies(logistic_loop canary_core)
