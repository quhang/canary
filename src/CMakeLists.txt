##
# Canary core library.

project(canary CXX)

# Initializes internal cmake utilities.
include(internal_utils)
config_compiler_and_linker()

# Adds header file search directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# canary_core library target.
cxx_static_library(canary_core "${cxx_default}"
    shared/initialize.cc
    shared/internal_flag.cc
    shared/internal_network.cc
    shared/partition_map.cc
    message/message.cc
    controller/controller_communication_manager.cc
    worker/worker_communication_manager.cc
    worker/worker_data_router.cc
    worker/worker_scheduler.cc
    )
add_dependencies(canary_core project_cereal)
# Library linking order matters. Dependency libraries are public so that built
# executables can link to them.
target_link_libraries(canary_core PUBLIC glog gflags event_pthreads event_core pthread dl)

# Worker executable.
cxx_executable(canary_worker worker "canary_core")
# Controller executable.
cxx_executable(canary_controller controller "canary_core")

file(GLOB_RECURSE ALL_CANARY_SOURCES "*.cc" "*.h")

# Checks code style.
add_custom_target(check "${CPPLINT_TOOL}" ${ALL_CANARY_SOURCES} COMMENT "Checks coding style.")

# Formats codes in place.
add_custom_target(format "${FORMAT_TOOL}" ${ALL_CANARY_SOURCES} COMMENT "Formats all source codes.")
