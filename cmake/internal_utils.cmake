# Defines functions and macros useful for building.

# Defines compiler/linker flags used for building.
macro(config_compiler_and_linker)
  if(not CMAKE_COMPILER_IS_GNUCXX)
    message(FATAL_ERROR "Compiling environment other than GNU gcc is not supported")
  endif()
  set(cxx_base_flags "-std=c++14 -Wall -Wshadow")
  set(cxx_strict_flags "-Wextra -Woverloaded-virtual")
  set(cxx_relaxed "${cxx_base_flags}")
  set(cxx_default "${cxx_base_flags} ${cxx_strict_flags}")
endmacro()

# Helper function for defining libraries.
function(cxx_library_with_type name type cxx_flags)
  add_library(${name} ${type} ${ARGN})
  set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cxx_flags}")
endfunction()

# Defines a shared library target.
# cxx_shared_library(library_name "${cxx_default}" src...)
function(cxx_shared_library name cxx_flags)
  cxx_library_with_type(${name} SHARED "${cxx_flags}" ${ARGN})
endfunction()

# Defines a static library target.
# cxx_static_library(library_name "${cxx_default}" src...)
function(cxx_static_library name cxx_flags)
  cxx_library_with_type(${name} STATIC "${cxx_flags}" ${ARGN})
endfunction()

# Helper function for defining executables.
function(cxx_executable_with_flags name cxx_flags libs)
  add_executable(${name} ${ARGN})
  if (cxx_flags)
    set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cxx_flags}")
  endif()
  foreach(lib ${libs})
    target_link_libraries(${name} ${lib})
  endforeach()
endfunction()

# Defines an executable target.
# cxx_executable(name dir "libs" srcs...)
function(cxx_executable name dir libs)
  cxx_executable_with_flags(
    ${name} "${cxx_default}" "${libs}" "${dir}/${name}.cc" ${ARGN})
endfunction()

# Helper function for defining tests.
function(cxx_test_with_flags name cxx_flags libs)
  cxx_executable_with_flags(${name} "${cxx_flags}" "${libs}" ${ARGN})
  add_test(${name} ${name})
endfunction()

# Defines a test target.
# cxx_test(name "libs" srcs...)
function(cxx_test name libs)
  cxx_test_with_flags("${name}" "${cxx_default}" "${libs}" "test/${name}.cc" ${ARGN})
endfunction()
