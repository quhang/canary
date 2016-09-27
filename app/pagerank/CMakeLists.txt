##
# Pagerank application running a fixed number of loops.

project(pagerank CXX)

# Initializes internal cmake utilities.
include(internal_utils)
config_compiler_and_linker()

# Adds header file search directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# An application is built as a shared library.
cxx_shared_library(pagerank "${cxx_default}" pagerank.cc)
# canary_core library must be built before.
add_dependencies(pagerank canary_core)
