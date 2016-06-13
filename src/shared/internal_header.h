/*
 * Copyright 2015 Stanford University.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither the name of the copyright holders nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @file src/shared/internal_header.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Internal header files.
 */

#ifndef CANARY_SRC_SHARED_INTERNAL_HEADER_H_
#define CANARY_SRC_SHARED_INTERNAL_HEADER_H_

// Gflags library for managing command line flags.
#include <gflags/gflags.h>
// Glog library for logging.
#include <glog/logging.h>

// C++ libraries.
#include <cinttypes>
#include <functional>
#include <sstream>
#include <type_traits>

/**
 * Marks a class as singleton which only offers static access.
 */
#ifndef SINGLETON_STATIC
#define SINGLETON_STATIC(T) \
  T() = delete;             \
  ~T() = delete
#endif  // SINGLETON_STATIC

/**
 * Marks a class as non-copyable and non-movable.
 */
#ifndef NON_COPYABLE_NOR_MOVABLE
#define NON_COPYABLE_NOR_MOVABLE(T) \
  T(const T&) = delete;             \
  T(T&&) = delete;                  \
  T& operator=(const T&) = delete;  \
  T& operator=(T&&) = delete
#endif  // NON_COPYABLE_NOR_MOVABLE

/**
 * Marks a class as non-copyable.
 */
#ifndef NON_COPYABLE
#define NON_COPYABLE(T) \
  T(const T&) = delete; \
  T& operator=(const T&) = delete
#endif  // NON_COPYABLE

#endif  // CANARY_SRC_SHARED_INTERNAL_HEADER_H_
