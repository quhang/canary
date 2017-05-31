##
# Sets up all dependency libraries.
#
# default_flags.cmake should be called before it.

# Imports ExternaProject module.
include(ExternalProject)

# Imports the threading library as Threads::Threads.
find_package(Threads REQUIRED)

# Sets build path for dependency libraries.
set(DEPENDENCY_BUILD_PATH ${CMAKE_BINARY_DIR}/dependency_src)
# Sets install path for dependency libraries.
set(DEPENDENCY_INSTALL_PATH ${CMAKE_BINARY_DIR}/dependency)

# Builds and installs gflags library.
ExternalProject_Add(project_gflags
  URL ${CMAKE_SOURCE_DIR}/packages/gflags_v2.1.2.tar.gz
  PREFIX ${DEPENDENCY_BUILD_PATH}/gflags
  CMAKE_ARGS -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX:PATH=${DEPENDENCY_INSTALL_PATH}
)
add_library(gflags STATIC IMPORTED)
add_dependencies(gflags project_gflags)
set_property(TARGET gflags PROPERTY IMPORTED_LOCATION ${DEPENDENCY_INSTALL_PATH}/lib/libgflags.a)

# Builds and installs glog library.
ExternalProject_Add(project_glog
  URL ${CMAKE_SOURCE_DIR}/packages/glog_v0.3.4.tar.gz
  # glog depends on gflags.
  DEPENDS project_gflags
  PREFIX ${DEPENDENCY_BUILD_PATH}/glog
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --with-gflags=${DEPENDENCY_INSTALL_PATH} --enable-shared=false --prefix=${DEPENDENCY_INSTALL_PATH}
  BUILD_COMMAND make
  INSTALL_COMMAND make install
)
add_library(glog STATIC IMPORTED)
add_dependencies(glog project_glog)
set_property(TARGET glog PROPERTY IMPORTED_LOCATION ${DEPENDENCY_INSTALL_PATH}/lib/libglog.a)

# Installs header-only cereal library.
ExternalProject_Add(project_cereal
  URL ${CMAKE_SOURCE_DIR}/packages/cereal_v1.1.2.tar.gz
  PREFIX ${DEPENDENCY_BUILD_PATH}/cereal
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND mkdir -p ${DEPENDENCY_INSTALL_PATH}/include/ && cp -r <SOURCE_DIR>/include/cereal/ ${DEPENDENCY_INSTALL_PATH}/include/
)

# Builds and installs libevent library.
ExternalProject_Add(project_libevent
  URL ${CMAKE_SOURCE_DIR}/packages/libevent_v2.1.5.tar.gz
  PREFIX ${DEPENDENCY_BUILD_PATH}/libevent
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --enable-shared=false --disable-openssl --prefix=${DEPENDENCY_INSTALL_PATH}
  BUILD_COMMAND make
  INSTALL_COMMAND make install
)
add_library(event_core STATIC IMPORTED)
add_dependencies(event_core project_libevent)
set_property(TARGET event_core PROPERTY IMPORTED_LOCATION ${DEPENDENCY_INSTALL_PATH}/lib/libevent_core.a)
add_library(event_pthreads STATIC IMPORTED)
add_dependencies(event_pthreads project_libevent)
set_property(TARGET event_pthreads PROPERTY IMPORTED_LOCATION ${DEPENDENCY_INSTALL_PATH}/lib/libevent_pthreads.a)

# Builds and installs gtest library.
ExternalProject_Add(project_gtest
  URL ${CMAKE_SOURCE_DIR}/packages/gtest_v1.7.0.tar.gz
  PREFIX ${DEPENDENCY_BUILD_PATH}/gtest
  CMAKE_ARGS -DCMAKE_BUILD_TYPE:STRING=Release
  # gtest does not allow "make install".
  INSTALL_COMMAND mkdir -p ${DEPENDENCY_INSTALL_PATH}/include/ && cp -r <SOURCE_DIR>/include/gtest/ ${DEPENDENCY_INSTALL_PATH}/include/
    && mkdir -p ${DEPENDENCY_INSTALL_PATH}/lib/ && cp -r <BINARY_DIR>/libgtest.a ${DEPENDENCY_INSTALL_PATH}/lib/
    && cp -r <BINARY_DIR>/libgtest_main.a ${DEPENDENCY_INSTALL_PATH}/lib/
)
add_library(gtest STATIC IMPORTED)
add_dependencies(gtest project_gtest)
set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${DEPENDENCY_INSTALL_PATH}/lib/libgtest.a)
add_library(gtest_main STATIC IMPORTED)
add_dependencies(gtest_main project_gtest)
set_property(TARGET gtest_main PROPERTY IMPORTED_LOCATION ${DEPENDENCY_INSTALL_PATH}/lib/libgtest_main.a)

IF(build_gpu)
  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -O3 --std=c++11 --gpu-architecture=compute_35 -Xcompiler -fPIC)

  # Installs header-only Eigen library.
  ExternalProject_Add(project_eigen
    URL ${CMAKE_SOURCE_DIR}/packages/eigen_v3.3.3.tar.gz
    PREFIX ${DEPENDENCY_BUILD_PATH}/eigen
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND mkdir -p ${DEPENDENCY_INSTALL_PATH}/include/ && cp -r <SOURCE_DIR>/Eigen ${DEPENDENCY_INSTALL_PATH}/include/ && cp -r <SOURCE_DIR>/unsupported ${DEPENDENCY_INSTALL_PATH}/include/
  )
ENDIF(build_gpu)
