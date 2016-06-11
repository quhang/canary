Canary Library
==============

Operating System
----------------

Canary library has only been tested under Ubuntu 14.04.

Dependency
----------

Cmake 3.2.2 is used for building management, and GNU C++ 5.3 is used for
compiling. High versions might work as well.

Install basic tools:
$ sudo apt-get install build-essential software-properties-common

Install Cmake:
$ sudo add-apt-repository ppa:george-edison55/cmake-3.x
$ sudo apt-get update
$ sudo apt-get install cmake cmake-curses-gui

Install GNU C++:
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
$ sudo apt-get update
$ sudo apt-get install g++-5

Configure GNU C++:
$ sudo update-alternatives --remove-all gcc
$ sudo update-alternatives --remove-all g++
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 100 --slave /usr/bin/g++ g++ /usr/bin/g++-5
$ sudo update-alternatives --config gcc
