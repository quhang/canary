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

cxx_executable(basic_server . "canary_core")
cxx_executable(basic_client . "canary_core")

# Test the command exchanging between one controller and multile workers.
cxx_executable(basic_command_exchange . "canary_core")

# Test the command exchanging between one controller and one worker.
cxx_executable(worker_command_manager worker "canary_core")
cxx_executable(controller_command_manager controller "canary_core")
