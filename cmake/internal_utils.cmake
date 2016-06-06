##
# Defines functions and macros useful for building Canary.

# Defines compiler/linker flags used for building.
#
# Implemented as a macro so as to introduce side effects.
macro(config_compiler_and_linker)
  if(NOT CMAKE_COMPILER_IS_GNUCXX)
    message(FATAL_ERROR "The compiler must be GNU C++.")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5)
      message(FATAL_ERROR "GNU C++ compiler version must be at least 5.")
    endif()
  endif()
  set(cxx_base_flags "-std=c++14 -Wall -Wshadow")
  set(cxx_strict_flags "-Wextra -Woverloaded-virtual")
  # Relaxed compiler flags.
  set(cxx_relaxed "${cxx_base_flags}")
  # Default compiler flags.
  set(cxx_default "${cxx_base_flags} ${cxx_strict_flags}")
endmacro()

# Helper function for defining a library target.
function(cxx_library_with_type name type cxx_flags)
  add_library(${name} ${type} ${ARGN})
  set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cxx_flags}")
endfunction()

# Defines a shared library target.
#
# Usage:
# cxx_shared_library(library_name "${cxx_default}" src...)
function(cxx_shared_library name cxx_flags)
  cxx_library_with_type(${name} SHARED "${cxx_flags}" ${ARGN})
endfunction()

# Defines a static library target.
#
# Usage:
# cxx_static_library(library_name "${cxx_default}" src...)
function(cxx_static_library name cxx_flags)
  cxx_library_with_type(${name} STATIC "${cxx_flags}" ${ARGN})
endfunction()

# Helper function for defining an executable target.
function(cxx_executable_with_flags name cxx_flags libs)
  add_executable(${name} ${ARGN})
  if (cxx_flags)
    set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cxx_flags}")
  endif()
  foreach(lib ${libs})
    target_link_libraries(${name} ${lib})
    add_dependencies(${name} ${lib})
  endforeach()
endfunction()

# Defines an executable target.
#
# Usage:
# cxx_executable(name dir "libs" srcs...)
function(cxx_executable name dir libs)
  cxx_executable_with_flags(
    ${name} "${cxx_default}" "${libs}" "${dir}/${name}.cc" ${ARGN})
endfunction()

# Helper function for defining a test.
function(cxx_test_with_flags name cxx_flags libs)
  cxx_executable_with_flags(${name} "${cxx_flags}" "${libs}" ${ARGN})
  add_test(${name} ${name})
endfunction()

# Defines a test target.
#
# Usage:
# cxx_test(name "libs" srcs...)
function(cxx_test name dir libs)
  cxx_test_with_flags("${name}" "${cxx_default}" "${libs}" "${dir}/${name}.cc" ${ARGN})
endfunction()
