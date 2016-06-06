project(canary_library_test CXX)

# Initializes internal cmake utilities.
include(internal_utils)
config_compiler_and_linker()

# Adds header file search directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/)

# Configure test libraries.
set(TEST_LIBRARIES canary_core gtest_main gtest)

cxx_test(basic worker "${TEST_LIBRARIES}")
