##
# Barrier test application.

project(cuda_sumup CXX)

include(internal_utils)
# Initializes internal cmake utilities.
config_compiler_and_linker()

# Adds header file search directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDA)

IF(CUDA_FOUND)
  set(CUDA_NVCC_FLAGS
      ${CUDA_NVCC_FLAGS}; -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -O3 --std=c++11 -Xcompiler -fPIC)
  # An application is built as a shared library.
  CUDA_ADD_LIBRARY(cuda_sumup cuda_sumup.cc cuda_helper.cu SHARED)
  set_target_properties(cuda_sumup PROPERTIES COMPILE_FLAGS "${cxx_default}")
  # canary_core library must be built before.
  add_dependencies(cuda_sumup canary_core)
ENDIF()
