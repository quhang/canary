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
  if (name == "default") {
    // Round robin
    return new OrderedEvenlyPlacementSchedule(scheduling_info);
  } else if (name == "consecutive") {
    // Place evenly like round robin, but consecutive on the same worker
    return new ConsecutiveEvenPlacementSchedule(scheduling_info);
  } else if (name == "application") {
    // Provided by application
    return new ApplicationPlacementSchedule(scheduling_info);
  } else {
    // Random
    return new EvenlyPlacementSchedule(scheduling_info);
  }
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
    if (iter != worker_map.end() &&
        iter->second.worker_state ==
            SchedulingInfo::WorkerRecord::WorkerState::RUNNING &&
        (last_assigned_partitions_ < iter->second.num_cores)) {
      worker_id = last_assigned_worker_id_;
      ++last_assigned_partitions_;
    } else {
      // Assigns a partition to the next running worker.
      do {
        worker_id = NextAssignWorkerId();
      } while (worker_map.at(worker_id).worker_state !=
               SchedulingInfo::WorkerRecord::WorkerState::RUNNING);
      last_assigned_partitions_ = 1;
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


void OrderedEvenlyPlacementSchedule::PlaceApplication(
    ApplicationId application_id) {
  const auto& application_record =
      scheduling_info_->get_application_map().at(application_id);
  const auto& variable_group_info_map =
      *application_record.variable_group_info_map;
  for (const auto& pair : variable_group_info_map) {
    std::vector<WorkerId> worker_id_list;
    next_assigned_worker_id_ = WorkerId::INVALID;
    for (int index = 0; index < pair.second.parallelism; ++index) {
      next_assigned_worker_id_ = NextAssignWorkerId();
      scheduling_info_->DecidePartitionPlacement(
          FullPartitionId{application_id, pair.first,
                          static_cast<PartitionId>(index)},
          next_assigned_worker_id_);
    }
  }
}

WorkerId OrderedEvenlyPlacementSchedule::NextAssignWorkerId() {
  const auto& worker_map = scheduling_info_->get_worker_map();
  CHECK(!worker_map.empty());
  auto iter = worker_map.upper_bound(next_assigned_worker_id_);
  if (iter == worker_map.end()) {
    next_assigned_worker_id_ = worker_map.begin()->first;
  } else {
    next_assigned_worker_id_ = iter->first;
  }
  return next_assigned_worker_id_;
}

void ConsecutiveEvenPlacementSchedule::PlaceApplication(
    ApplicationId application_id) {
  const auto& application_record =
      scheduling_info_->get_application_map().at(application_id);
  const auto& variable_group_info_map =
      *application_record.variable_group_info_map;
  // for each variable
  for (const auto& pair : variable_group_info_map) {
    std::vector<WorkerId> worker_id_list;
    // get worker assignment for each partition
    GetWorkerAssignment(pair.second.parallelism, &worker_id_list);
    // invoke api function for this assignment, for each partition
    auto iter = worker_id_list.begin();
    for (int index = 0; index < pair.second.parallelism; ++index) {
      scheduling_info_->DecidePartitionPlacement(
          FullPartitionId{application_id, pair.first,
                          static_cast<PartitionId>(index)},
          *(iter++));
    }  // for each partition
  } // for each variable
}

void ConsecutiveEvenPlacementSchedule::GetWorkerAssignment(
    int num_slot, std::vector<WorkerId>* assignment) {
  assignment->clear();
  const auto& worker_map = scheduling_info_->get_worker_map();
  // count number of running workers and make a list of the worker ids
  int num_workers = 0;
  std::vector<WorkerId> worker_list;
  for (auto pair : worker_map) {
    if (pair.second.worker_state ==
        SchedulingInfo::WorkerRecord::WorkerState::RUNNING) {
      num_workers++;
      worker_list.push_back(pair.first);
    }
  }
  // number of partitions to place on each worker
  if (num_slot == 1) {
    // reduction variables
    assignment->push_back(worker_list[0]);
  }
  int partitions_per_worker = num_slot/num_workers;
  for (int w = 0; w < num_workers; ++w) {
    for (int p = 0; p < partitions_per_worker; ++p) {
      // consecutive partition ranks have the same worker
      assignment->push_back(worker_list[w]);
    }  // for p
  }  // for w
}  // GetWorkerAssignment

void ApplicationPlacementSchedule::PlaceApplication(
    ApplicationId application_id) {
  const auto& application_record =
      scheduling_info_->get_application_map().at(application_id);
  const auto& variable_group_info_map =
      *application_record.variable_group_info_map;
  const auto& initial_group_placement =
      *application_record.initial_variable_group_placement;
  // Make a list of all running workers.
  const auto& worker_map = scheduling_info_->get_worker_map();
  std::vector<WorkerId> running_workers;
  for (auto pair : worker_map) {
    if (pair.second.worker_state ==
        SchedulingInfo::WorkerRecord::WorkerState::RUNNING) {
      running_workers.push_back(pair.first);
    }  // if running
  }  // for pair in worker_map
  // for each variable
  for (const auto& pair : variable_group_info_map) {
    std::vector<WorkerId> assigned_workers;
    // Get worker assignment for each grop.
    const std::vector<int> &initial_placement =
      initial_group_placement.at(pair.first);
    CHECK_EQ(int(initial_placement.size()), pair.second.parallelism) <<
      "Size of placement provided by application != number of partitions";
    GetWorkerAssignment(running_workers, initial_placement, &assigned_workers);
    // Invoke api function for this assignment, for each partition.
    auto iter = assigned_workers.begin();
    for (int index = 0; index < pair.second.parallelism; ++index) {
      std::cout << "Initial placement: " << int(pair.first) << ", " << index
                << ", " << int(*iter) << std::endl;
      scheduling_info_->DecidePartitionPlacement(
          FullPartitionId{application_id, pair.first,
                          static_cast<PartitionId>(index)},
          *(iter++));
    }  // for each partition
  } // for each variable
}

void ApplicationPlacementSchedule::GetWorkerAssignment(
  const std::vector<WorkerId> &running_workers,
  const std::vector<int> &initial_placement,
  std::vector<WorkerId> *assigned_workers) {
  assigned_workers->clear();
  for (size_t i = 0; i < initial_placement.size(); ++i) {
    int w = initial_placement[i];
    CHECK_GE(w, 0);
    CHECK_LT(w, int(running_workers.size()));
    assigned_workers->push_back(running_workers[w]);
  }  // for i < initial_placement.size()
}  // GetWorkerAssignment

}  // namespace canary
