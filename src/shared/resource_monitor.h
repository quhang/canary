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
 * @file src/shared/resource_monitor.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class ResourceMonitor.
 */

#ifndef CANARY_SRC_SHARED_RESOURCE_MONITOR_H_
#define CANARY_SRC_SHARED_RESOURCE_MONITOR_H_

#include <regex>

#include "canary/canary_internal.h"

namespace canary {

/**
 * Monitors the resource usage of a worker, i.e. CPU usage, and memory usage.
 * Memory usage monitor has not been added yet.
 */
class ResourceMonitor {
 public:
  //! Constructor.
  ResourceMonitor() {
    PCHECK(pthread_mutex_init(&internal_lock_, nullptr) == 0);
  }
  //! Destructor.
  virtual ~ResourceMonitor() { pthread_mutex_destroy(&internal_lock_); }
  //! Initializes the monitor and spawns the background monitoring thread.
  void Initialize();
  //! Gets the cpu usage percentage of all local processes.
  double get_all_cpu_usage_percentage() const {
    CHECK(initialized_);
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    const double result = all_cpu_usage_percentage_;
    PCHECK(pthread_mutex_unlock(&internal_lock_) == 0);
    return result;
  }
  //! Gets the cpu usage percentage of Canary.
  double get_canary_cpu_usage_percentage() const {
    CHECK(initialized_);
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    const double result = canary_cpu_usage_percentage_;
    PCHECK(pthread_mutex_unlock(&internal_lock_) == 0);
    return result;
  }
  //! Gets the available memory in GBs.
  double get_available_memory_gb() const { return 0; }
  //! Gets the used memory in GBs.
  double get_used_memory_gb() const { return 0; }
  //! Measures the resource usage, and returns true if it succeeds.
  bool Measure();

 private:
  template <typename IteratorType>
  bool GetNextUnsignedLong(IteratorType* iterator, unsigned long* result);
  void SubmitMeasurement(unsigned long canary_cpu_ticks,
                         unsigned long all_cpu_ticks,
                         unsigned long idle_cpu_ticks);

 private:
  bool initialized_ = false;
  std::regex token_regex_ = std::regex("\\s+");

  bool after_first_sample_ = false;
  unsigned long last_canary_cpu_ticks_;
  unsigned long last_all_cpu_ticks_;
  unsigned long last_idle_cpu_ticks_;
  double canary_cpu_usage_percentage_ = 0;
  double all_cpu_usage_percentage_ = 0;

  mutable pthread_mutex_t internal_lock_;
};

}  // namespace canary
#endif  // CANARY_SRC_SHARED_RESOURCE_MONITOR_H_
