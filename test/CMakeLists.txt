project(canary_library_test CXX)

# Initializes internal cmake utilities.
include(internal_utils)
config_compiler_and_linker()

# Adds header file search directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/)

# Configure test libraries.
set(TEST_LIBRARIES canary_core gtest)

cxx_test(basic_marshal . "${TEST_LIBRARIES}")
cxx_test(intensive_marshal . "${TEST_LIBRARIES}")

# Test the command exchanging between one controller and multile workers.
cxx_test(basic_command_exchange . "${TEST_LIBRARIES}")

# Test the data exchanging between multile workers.
cxx_executable(basic_data_exchange . "canary_core")

cxx_executable(basic_server . "canary_core")
cxx_executable(basic_client . "canary_core")

cxx_executable(basic_load_and_print . "canary_core")

cxx_executable(playground . "canary_core")
