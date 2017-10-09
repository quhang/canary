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
    return new BalancedPartitionNumberLoadSchedule(scheduling_info);
  }
  if (name == "test") {
    return new TestLoadSchedule(scheduling_info);
  }
  if (name == "move") {
    return new TestMoveLoadSchedule(scheduling_info);
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

void TestMoveLoadSchedule::BalanceLoad() {
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
  LOG(INFO) << "Move from worker(" << get_value(first_iter->first)
            << ") to worker(" << get_value(second_iter->first) << ").";
  LOG(INFO) << "Move Partition("
            << get_value(first_full_partition_id.application_id) << "/"
            << get_value(first_full_partition_id.variable_group_id) << "/"
            << get_value(first_full_partition_id.partition_id)
            << ")";
  scheduling_info_->DecidePartitionPlacement(first_full_partition_id,
                                             second_iter->first);
}

void BalancedPartitionNumberLoadSchedule::BalanceLoad() {
  GrabWorkerInfo();
  if (total_cores_ == 0) {
    return;
  }
  CalculateExpectedNumPartitions();
  IssuePartitionPlacementDecision();
}

void BalancedPartitionNumberLoadSchedule::GrabWorkerInfo() {
  worker_info_.clear();
  total_cores_ = 0;
  total_partitions_ = 0;
  for (const auto& pair : scheduling_info_->get_worker_map()) {
    auto& worker_info_record = worker_info_[pair.first];
    worker_info_record.worker_id = pair.first;
    for (const auto& internal_pair : pair.second.owned_partitions) {
      const auto& partition_set = internal_pair.second;
      worker_info_record.owned_partitions.insert(partition_set.begin(),
                                                 partition_set.end());
      total_partitions_ += partition_set.size();
    }
    if (pair.second.worker_state ==
        SchedulingInfo::WorkerRecord::WorkerState::RUNNING) {
      worker_info_record.num_cores = pair.second.num_cores;
      total_cores_ += pair.second.num_cores;
    } else {
      worker_info_record.num_cores = 0;
    }
  }
}

void BalancedPartitionNumberLoadSchedule::CalculateExpectedNumPartitions() {
  const int num_partitions_per_core = total_partitions_ / total_cores_;
  int num_remain_partitions = total_partitions_ % total_cores_;
  for (auto& pair : worker_info_) {
    auto& worker_info_record = pair.second;
    if (worker_info_record.num_cores == 0) {
      worker_info_record.expected_num_partitions = 0;
    } else {
      worker_info_record.expected_num_partitions =
          worker_info_record.num_cores * num_partitions_per_core;
    }
  }
  while (num_remain_partitions != 0) {
    for (auto& pair : worker_info_) {
      auto& worker_info_record = pair.second;
      if (worker_info_record.num_cores != 0) {
        ++worker_info_record.expected_num_partitions;
        if (--num_remain_partitions == 0) {
          break;
        }
      }
    }
  }
}

void BalancedPartitionNumberLoadSchedule::IssuePartitionPlacementDecision() {
  for (auto& pair : worker_info_) {
    auto& worker_info_record = pair.second;
    while (static_cast<int>(worker_info_record.owned_partitions.size()) >
           worker_info_record.expected_num_partitions) {
      OffloadPartition(&worker_info_record);
    }
  }
}

bool BalancedPartitionNumberLoadSchedule::OffloadPartition(
    WorkerInfoRecord* worker_info_record) {
  for (auto& pair : worker_info_) {
    auto& candidate_record = pair.second;
    if (static_cast<int>(candidate_record.owned_partitions.size()) <
        candidate_record.expected_num_partitions) {
      auto iter = worker_info_record->owned_partitions.begin();
      scheduling_info_->DecidePartitionPlacement(*iter,
                                                 candidate_record.worker_id);
      candidate_record.owned_partitions.insert(*iter);
      worker_info_record->owned_partitions.erase(iter);
      return true;
    }
  }
  return false;
}

void StragglerMitigationLoadSchedule::BalanceLoad() {
  GrabWorkerInfo();
  if (total_cores_ == 0) {
    return;
  }
  FigureOutStraggler();
  CalculateExpectedNumPartitions();
  IssuePartitionPlacementDecision();
}

void StragglerMitigationLoadSchedule::FigureOutStraggler() {
  double max_stolen_cpu = 0;
  WorkerId max_stolen_cpu_worker_id = WorkerId::INVALID;
  for (auto& pair : worker_info_) {
    auto& worker_record = scheduling_info_->get_worker_map().at(pair.first);
    double stolen_cpu = worker_record.all_cpu_usage_percentage -
                        worker_record.canary_cpu_usage_percentage;
    if (max_stolen_cpu_worker_id == WorkerId::INVALID ||
        stolen_cpu > max_stolen_cpu) {
      max_stolen_cpu = stolen_cpu;
      max_stolen_cpu_worker_id = pair.first;
    }
  }
  if (max_stolen_cpu_worker_id != WorkerId::INVALID) {
    total_cores_ -= worker_info_.at(max_stolen_cpu_worker_id).num_cores;
    worker_info_.at(max_stolen_cpu_worker_id).num_cores = 0;
    LOG(INFO) << "Offloading computations away from worker (id="
              << get_value(max_stolen_cpu_worker_id) << ").";
  }
}

}  // namespace canary
