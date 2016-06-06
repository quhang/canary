project(canary CXX)

include(internal_utils)
config_compiler_and_linker()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

cxx_static_library(canary_core "${cxx_default}" shared/internal.cc)
add_dependencies(canary_core project_gflags project_glog project_cereal project_libevent)
target_link_libraries(canary_core PUBLIC glog gflags event_pthreads event_core pthread dl)

cxx_executable(canary_worker worker "canary_core")
cxx_executable(canary_controller controller "canary_core")

file(GLOB_RECURSE ALL_SOURCES "*.cc" "*.h")

# Checks code style.
add_custom_target(check ${CMAKE_SOURCE_DIR}/tools/cpplint.py ${ALL_SOURCES} COMMENT "Check coding style.")

# Format codes in place.
add_custom_target(format clang-format-3.6 --style=Google -i ${ALL_SOURCES} COMMENT "Reformat all source codes.")
