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
 * @file src/shared/resource_monitor.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class ResourceMonitor.
 */

#include "shared/resource_monitor.h"

#include <string>
#include <thread>

namespace canary {

void ResourceMonitor::Initialize() {
  CHECK(!initialized_);
  initialized_ = true;
  std::thread measure_thread([this]() {
    while (true) {
      this->Measure();
      // Measures every 5 seconds.
      sleep(5);
    }
  });
  measure_thread.detach();
}

template <typename IteratorType>
bool ResourceMonitor::GetNextUnsignedLong(IteratorType* iterator,
                                          uint64_t* result) try {
  if (*iterator == IteratorType()) {
    return false;
  }
  if (++(*iterator) == IteratorType()) {
    return false;
  }
  if (result) {
    *result = std::stoul(*(*iterator));
  }
  return true;
} catch (std::exception&) {
  return false;
}

bool ResourceMonitor::Measure() {
  FILE* process_stat_file = fopen("/proc/self/stat", "r");
  if (!process_stat_file) {
    LOG(ERROR) << "Cannot open /proc/self/stat!";
    return false;
  }
  uint64_t canary_user_ticks, canary_sys_ticks;
  if (fscanf(process_stat_file,
             "%*d %*s %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu",
             &canary_user_ticks, &canary_sys_ticks) != 2) {
    LOG(ERROR) << "Cannot parse /proc/self/stat!";
    return false;
  }
  fclose(process_stat_file);
  FILE* global_stat_file = fopen("/proc/stat", "r");
  if (!global_stat_file) {
    LOG(ERROR) << "Cannot open /proc/stat!";
    return false;
  }
  constexpr int kLineBufferSize = 0x400;
  char line_buffer[kLineBufferSize];
  uint64_t global_user_ticks, global_sys_ticks, global_idle_ticks;
  while (fgets(line_buffer, kLineBufferSize, global_stat_file)) {
    std::string line_string(line_buffer);
    auto iter = std::sregex_token_iterator(line_string.begin(),
                                           line_string.end(), token_regex_, -1);
    if (iter == std::sregex_token_iterator() || *iter != "cpu") {
      continue;
    }
    if (!GetNextUnsignedLong(&iter, &global_user_ticks)) {
      fclose(global_stat_file);
      LOG(ERROR) << "Cannot parse /proc/stat!";
      return false;
    }
    if (!GetNextUnsignedLong(&iter, nullptr)) {
      fclose(global_stat_file);
      LOG(ERROR) << "Cannot parse /proc/stat!";
      return false;
    }
    if (!GetNextUnsignedLong(&iter, &global_sys_ticks)) {
      fclose(global_stat_file);
      LOG(ERROR) << "Cannot parse /proc/stat!";
      return false;
    }
    if (!GetNextUnsignedLong(&iter, &global_idle_ticks)) {
      fclose(global_stat_file);
      LOG(ERROR) << "Cannot parse /proc/stat!";
      return false;
    }
    SubmitMeasurement(canary_user_ticks + canary_sys_ticks,
                      global_user_ticks + global_sys_ticks, global_idle_ticks);
    fclose(global_stat_file);
    return true;
  }
  fclose(global_stat_file);
  LOG(ERROR) << "Cannot parse /proc/stat!";
  return false;
}

void ResourceMonitor::SubmitMeasurement(uint64_t canary_cpu_ticks,
                                        uint64_t all_cpu_ticks,
                                        uint64_t idle_cpu_ticks) {
  PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
  if (after_first_sample_) {
    canary_cpu_usage_percentage_ = 100. *
                                   (canary_cpu_ticks - last_canary_cpu_ticks_) /
                                   (all_cpu_ticks - last_all_cpu_ticks_ +
                                    idle_cpu_ticks - last_idle_cpu_ticks_);
    all_cpu_usage_percentage_ = 100. * (all_cpu_ticks - last_all_cpu_ticks_) /
                                (all_cpu_ticks - last_all_cpu_ticks_ +
                                 idle_cpu_ticks - last_idle_cpu_ticks_);
  } else {
    after_first_sample_ = true;
  }
  last_canary_cpu_ticks_ = canary_cpu_ticks;
  last_all_cpu_ticks_ = all_cpu_ticks;
  last_idle_cpu_ticks_ = idle_cpu_ticks;
  PCHECK(pthread_mutex_unlock(&internal_lock_) == 0);
}

}  // namespace canary
