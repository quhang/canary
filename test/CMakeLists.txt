project(parsec CXX)

include(internal_utils)
config_compiler_and_linker()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/)

set(test_libraries canary_core gtest_main gtest)
cxx_test(basic worker "${test_libraries}")
