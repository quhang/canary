project(parsec CXX)

include(internal_utils)
config_compiler_and_linker()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

cxx_shared_library(parsec "${cxx_default}" f.cc)
