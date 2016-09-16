##
# Top-level cmake file.

# Specifies the minimum required cmake version.
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# Global project name.
project(canary-distribution)

# Enables testing.
enable_testing()

# Adds cmake module search path.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Sets up default environment variables.
include(default_flags)

# Sets up dependency libraries.
include(build_dependency_library)

# Specifies header file directories.
include_directories(${DEPENDENCY_INSTALL_PATH}/include/ ${CMAKE_SOURCE_DIR}/include/)

# Sets up cpplint tool.
set(CPPLINT_TOOL ${CMAKE_SOURCE_DIR}/tools/cpplint.py)

# Sets up cpplint tool.
set(FORMAT_TOOL ${CMAKE_SOURCE_DIR}/tools/cppformat.sh)

# Builds canary library.
add_subdirectory(src)
# Builds applications.
add_subdirectory(app)
# Builds canary library tests.
add_subdirectory(test)
# Builds canary external headers.
add_subdirectory(include)

# Doxygen documentation.
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)
