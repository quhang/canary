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
include_directories(${DEPENDENCY_INSTALL_PATH}/include)

add_subdirectory(src)
