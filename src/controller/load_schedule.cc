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
 * @file src/controller/load_schedule.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class LoadSchedule.
 */

#include "controller/load_schedule.h"

#include <iterator>

namespace canary {

LoadSchedule* LoadSchedule::ConstructLoadSchedule(
    SchedulingInfo* scheduling_info, const std::string& name) {
  if (name == "default") {
    return new TestLoadSchedule(scheduling_info);
  }
  if (name == "test") {
    return new TestLoadSchedule(scheduling_info);
  }
  if (name == "balance") {
    return new BalancedPartitionNumberLoadSchedule(scheduling_info);
  }
  if (name == "straggler") {
    return new StragglerMitigationLoadSchedule(scheduling_info);
  }
  return nullptr;
}

void TestLoadSchedule::BalanceLoad() {
  const auto& worker_map = scheduling_info_->get_worker_map();
  if (worker_map.size() < 2u) {
    return;
  }
  const auto first_iter = worker_map.cbegin();
  if (first_iter->second.owned_partitions.empty()) {
    return;
  }
  const auto first_full_partition_id =
      *first_iter->second.owned_partitions.cbegin()->second.cbegin();
  const auto second_iter = std::next(first_iter);
  if (second_iter->second.owned_partitions.empty()) {
    return;
  }
  const auto second_full_partition_id =
      *second_iter->second.owned_partitions.cbegin()->second.cbegin();
  LOG(INFO) << "Swapping between worker(" << get_value(first_iter->first)
            << ") and worker(" << get_value(second_iter->first) << ").";
  LOG(INFO) << "Swapping Partition("
            << get_value(first_full_partition_id.application_id) << "/"
            << get_value(first_full_partition_id.variable_group_id) << "/"
            << get_value(first_full_partition_id.partition_id)
            << ") and Partition("
            << get_value(second_full_partition_id.application_id) << "/"
            << get_value(second_full_partition_id.variable_group_id) << "/"
            << get_value(second_full_partition_id.partition_id) << ").";
  scheduling_info_->DecidePartitionPlacement(first_full_partition_id,
                                             second_iter->first);
  scheduling_info_->DecidePartitionPlacement(second_full_partition_id,
                                             first_iter->first);
}

}  // namespace canary
