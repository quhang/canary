##
# Logistic loop application running a fixed number of loops.

project(cuda_logistic_loop CXX)

include(internal_utils)
# Initializes internal cmake utilities.
config_compiler_and_linker()

# Adds header file search directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# An application is built as a shared library.
cuda_add_library(cuda_logistic_loop cuda_logistic_loop.cc cuda_helper.cu SHARED)
set_target_properties(cuda_logistic_loop PROPERTIES COMPILE_FLAGS "${cxx_default}")
# canary_core library must be built before.
add_dependencies(cuda_logistic_loop canary_core)
