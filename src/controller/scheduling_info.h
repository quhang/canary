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
 * @file src/controller/scheduling_info.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class SchedulingInfo.
 */

#ifndef CANARY_SRC_CONTROLLER_SCHEDULING_INFO_H_
#define CANARY_SRC_CONTROLLER_SCHEDULING_INFO_H_

#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "shared/canary_internal.h"

#include "shared/canary_application.h"
#include "shared/partition_map.h"

namespace canary {

// Forward declaration.
class ControllerScheduler;

/**
 * A class that wraps the information as the input of a scheduling algorithm,
 * and also serves as the buffer of the output of a scheduling algorithm.
 */
class SchedulingInfo {
 public:
  //! Represents a worker.
  struct WorkerRecord {
    //! The number of cores.
    int num_cores = -1;
    //! CPU utilization percentage of all applications (might not be Canary).
    double all_cpu_usage_percentage = -1;
    //! CPU utilization percentage of Canary.
    double canary_cpu_usage_percentage = -1;
    //! All available memory space in GB.
    double available_memory_gb = -1;
    //! Memory space used by Canary in GB.
    double used_memory_gb = -1;
    //! Partitions owned by the worker.
    std::map<ApplicationId, std::set<FullPartitionId>> owned_partitions;
    // The number of owned partitions.
    int num_owned_partitions = 0;
    //! The worker's state.
    enum class WorkerState {
      INVALID,
      RUNNING,
      KILLED
    } worker_state = WorkerState::INVALID;

   private:
    friend class ControllerScheduler;
    //! Applications loaded by the worker.
    std::set<ApplicationId> loaded_applications;
  };

  //! Represents an application.
  struct ApplicationRecord {
    //! Describing the variables in the application.
    const CanaryApplication::VariableGroupInfoMap* variable_group_info_map =
        nullptr;
    //! The application's partition map.
    PerApplicationPartitionMap per_app_partition_map;
    //! Priority level of the application, lower means higher priority.
    PriorityLevel priority_level;
    //! The total of cycles spent for the application.
    double total_used_cycles = 0;
    //! Represents the execution state of an application.
    enum class ApplicationState {
      INVALID,
      RUNNING,
      AT_BARRIER,
      COMPLETE
    } application_state = ApplicationState::INVALID;

   private:
    friend class ControllerScheduler;
    //! The loaded application.
    CanaryApplication* loaded_application = nullptr;
    //! Internal usage, the dynamic loading handle of the application.
    void* loading_handle = nullptr;

    //! Binary location of the application.
    std::string binary_location;
    //! The application parameter.
    std::string application_parameter;
    //! The first barrier stage, at which all partitions should pause and wait
    // for resuming.
    StageId next_barrier_stage = StageId::INVALID;

    //! The total number of partitions.
    int total_partition = 0;
    //! The total number of complete partitions.
    int complete_partition = 0;
    //! The total number of partitions blocked at a barrier.
    int blocked_partition = 0;

    //! The identifier of a triggered report.
    int report_id = -1;
    //! The number of partitions that have reported since the last time the
    // report id was changed.
    std::set<FullPartitionId> report_partition_set;
    //! Lauchers that wait for reporting the running stats.
    std::vector<LaunchCommandId> report_command_list;
  };

  //! Represents a partition.
  struct PartitionRecord {
    //! Total cycles used by the partition.
    double total_used_cycles = 0;
    //! Cycles used by the partition in the recent iterations.
    std::list<double> loop_cycles;
    //! Maximum size of the list.
    static const int kMaxListSize = 10;
    //! The partition's state.
    enum class PartitionState {
      INVALID
    } partition_state = PartitionState::INVALID;
  };

 public:
  /*
   * Interfaces for accesssing scheduling information.
   */
  virtual const std::map<WorkerId, WorkerRecord>& get_worker_map() const = 0;
  virtual const std::map<ApplicationId, ApplicationRecord>&
  get_application_map() const = 0;
  virtual const std::map<FullPartitionId, PartitionRecord>&
  get_partition_record_map() const = 0;

  /*
   * Interfaces for specifying partition placement decisions.
   */
  void ClearPartitionPlacement() { partition_placement_decision_.clear(); }
  void DecidePartitionPlacement(FullPartitionId full_partition_id,
                                WorkerId to_worker_id) {
    partition_placement_decision_[full_partition_id] = to_worker_id;
  }
  std::map<FullPartitionId, WorkerId> RetrievePartitionPlacement() {
    return std::move(partition_placement_decision_);
  }

 private:
  std::map<FullPartitionId, WorkerId> partition_placement_decision_;
};

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_SCHEDULING_INFO_H_
