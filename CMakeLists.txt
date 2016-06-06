# Minimum required cmake version.
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# The global project name.
project(canary-distribution)

enable_testing()

# Adds cmake module search path.
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Changes default flags.
include(default_flags)

# Builds dependency libraries.
include(build_dependency_library)

# Specifies header file directory for dependency libraries.
include_directories(${DEPENDENCY_INSTALL_PATH}/include/ ${CMAKE_SOURCE_DIR}/include/)

# Canary library.
add_subdirectory(src)
# Applications.
add_subdirectory(apps)
# Canary tests.
add_subdirectory(test)
# Canary external headers.
add_subdirectory(include)
