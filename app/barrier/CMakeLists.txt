##
# Barrier test application.

project(barrier CXX)

# Initializes internal cmake utilities.
include(internal_utils)
config_compiler_and_linker()

# Adds header file search directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# An application is built as a shared library.
cxx_shared_library(barrier "${cxx_default}" barrier.cc)
# canary_core library must be built before.
add_dependencies(barrier canary_core)
