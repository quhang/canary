##
# Barrier test application.

project(cuda_sumup CXX)

include(internal_utils)
# Initializes internal cmake utilities.
config_compiler_and_linker()

# Adds header file search directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# An application is built as a shared library.
cuda_add_library(cuda_sumup cuda_sumup.cc cuda_helper.cu SHARED)
set_target_properties(cuda_sumup PROPERTIES COMPILE_FLAGS "${cxx_default}")
# canary_core library must be built before.
add_dependencies(cuda_sumup canary_core)
