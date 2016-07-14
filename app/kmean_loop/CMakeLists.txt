##
# Logistic regression application.

project(kmean_loop CXX)

# Initializes internal cmake utilities.
include(internal_utils)
config_compiler_and_linker()

# Adds header file search directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# An application is built as a shared library.
cxx_shared_library(kmean_loop "${cxx_default}" kmean_loop.cc)
# canary_core library must be built before.
add_dependencies(kmean_loop canary_core)
