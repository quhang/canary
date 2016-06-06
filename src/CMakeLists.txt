project(canary CXX)

add_executable(canary-hello main.cc)
add_dependencies(canary-hello project_gflags project_glog project_cereal project_libevent project_gtest)
target_link_libraries(canary-hello glog gflags event_pthreads event_core pthread)
