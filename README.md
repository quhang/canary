Canary: A Cloud Compute Engine Optimized for Computational Workloads
====================================================================

Building
--------

The recommended building environment for Canary is Ubuntu 16.04, using GNU C++
v5.4 or higher, and CMake v3.5 or higher. These building tools can be installed
with:

> sudo apt-get install build-essential cmake

Earlier Ubuntu distributions might also work with CMake v3.1+ and a C++
compiler that supports C++14 standard. Installing these building tools might
require manually adding software sources:

> sudo apt-get install software-properties-common

> sudo add-apt-repository ppa:george-edison55/cmake-3.x

> sudo apt-get update

> sudo apt-get install cmake

> sudo add-apt-repository ppa:ubuntu-toolchain-r/test

> sudo apt-get update

> sudo apt-get install g++-5

Canary uses CMake as the building management tool. To build Canary, go to the
top Canary source directory, and type:

> mkdir build

> cd build

> cmake ../

> make

External dependency libraries are incorporated into the code base, and will be
compiled automatically. The libraries are installed under the *build/*
directory, and only used by Canary. Following libraries are used: Cereal,
libevent, GFlags, GLog, GTest.

Running Canary
--------------

Launching a Canary cluster on the local machine requires launching one
controller process and at least one worker process:

> ./build/src/canary_controller

> ./build/src/canary_worker

Then, use the launcher binary to start an application:

> ./build/src/canary_launcher --launch_application=./app/logistic_loop/liblogistic_loop.so

It will execute a logistic regression application which outputs the result.

Launching multiple worker processes on the same server requires using different
ports for them. Many options are also available to tune the behavior of the
controller and workers: passing *--help* after a binary executable to explore
those options. The launcher binary can be used to control the lifetime of an
application, for example, pausing it.

Writing Applications
--------------------

All the header files required to write a Canary application are included in
*include/* directory. Each application is complied into a dynamic library, and
a Canary cluster can run multiple applications by loading applications
dynamically. Several sample applications are included in *app/* directory.

Writing Scheduling Algorithms
-----------------------------

Other than using the launcher to control the behavior of workers, a user can
write scheduling algorithms to decide how to place data partitions on workers.
Inherenting from *src/controller/placement_schedule.h* to decide how to place
partitions when an application starts. Inherenting from
*src/controller/load_schedule.h* to decide how to migrate partitions during
runtime.
