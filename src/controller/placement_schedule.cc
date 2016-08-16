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
 * @file src/controller/placement_schedule.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class PlacementSchedule.
 */

#include "controller/placement_schedule.h"

#include <algorithm>
#include <vector>

namespace canary {

PlacementSchedule* PlacementSchedule::ConstructPlacementSchedule(
    SchedulingInfo* scheduling_info, const std::string& name) {
  CHECK_EQ(name, "default") << "Placement algorithm is not implemented!";
  return new EvenlyPlacementSchedule(scheduling_info);
}

void EvenlyPlacementSchedule::PlaceApplication(ApplicationId application_id) {
  const auto& application_record =
      scheduling_info_->get_application_map().at(application_id);
  const auto& variable_group_info_map =
      *application_record.variable_group_info_map;
  for (const auto& pair : variable_group_info_map) {
    // Birdshot randomized placement.
    // Assigns partitions to workers randomly, and the number of assigned
    // partitions is based on the number of worker cores.
    std::vector<WorkerId> worker_id_list;
    GetWorkerAssignment(pair.second.parallelism, &worker_id_list);
    std::random_shuffle(worker_id_list.begin(), worker_id_list.end());
    auto iter = worker_id_list.begin();
    for (int index = 0; index < pair.second.parallelism; ++index) {
      scheduling_info_->DecidePartitionPlacement(
          FullPartitionId{application_id, pair.first,
                          static_cast<PartitionId>(index)},
          *(iter++));
    }
  }
}

void EvenlyPlacementSchedule::GetWorkerAssignment(
    int num_slot, std::vector<WorkerId>* assignment) {
  assignment->clear();
  assignment->resize(num_slot);
  const auto& worker_map = scheduling_info_->get_worker_map();
  for (auto& worker_id : *assignment) {
    auto iter = worker_map.find(last_assigned_worker_id_);
    if (iter == worker_map.end()) {
      // Assigns a partition to the next worker.
      worker_id = NextAssignWorkerId();
      last_assigned_partitions_ = 1;
      continue;
    }
    // TODO(quhang): unclear.
    if (iter->second.num_cores == -1) {
      LOG(WARNING) << "The number of cores for a worker is unknown!";
      CHECK_EQ(last_assigned_partitions_, 1);
      worker_id = NextAssignWorkerId();
      last_assigned_partitions_ = 1;
    } else {
      if (last_assigned_partitions_ < iter->second.num_cores) {
        worker_id = last_assigned_worker_id_;
        ++last_assigned_partitions_;
      } else {
        worker_id = NextAssignWorkerId();
        last_assigned_partitions_ = 1;
      }
    }
  }
}

WorkerId EvenlyPlacementSchedule::NextAssignWorkerId() {
  const auto& worker_map = scheduling_info_->get_worker_map();
  CHECK(!worker_map.empty());
  auto iter = worker_map.upper_bound(last_assigned_worker_id_);
  if (iter == worker_map.end()) {
    last_assigned_worker_id_ = worker_map.begin()->first;
  } else {
    last_assigned_worker_id_ = iter->first;
  }
  return last_assigned_worker_id_;
}

}  // namespace canary
