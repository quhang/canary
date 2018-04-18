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
 * @file src/message/command_message.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class CommandMessage.
 */

#ifndef CANARY_SRC_MESSAGE_COMMAND_MESSAGE_H_
#define CANARY_SRC_MESSAGE_COMMAND_MESSAGE_H_

#include <list>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include "shared/canary_internal.h"

#include "message/message.h"

namespace canary {
namespace message {

//! Used for testing.
struct TestWorkerCommand {
  std::string test_string;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(test_string);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, TEST_WORKER_COMMAND, TestWorkerCommand);

//! Used for testing.
struct TestControllerCommand {
  WorkerId from_worker_id;
  std::string test_string;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id, test_string);
  }
};
REGISTER_MESSAGE(CONTROLLER_COMMAND, TEST_CONTROLLER_COMMAND,
                 TestControllerCommand);

/*
 * Controller to worker commands.
 */

typedef std::list<std::pair<VariableGroupId, PartitionId>> PartitionList;

//! Loads an application, and saves its booting information at the worker
// scheduler.
struct WorkerLoadApplication {
  ApplicationId application_id;
  std::string binary_location;
  std::string application_parameter;
  StageId first_barrier_stage;
  PriorityLevel priority_level;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, binary_location, application_parameter,
            first_barrier_stage, priority_level);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_LOAD_APPLICATION,
                 WorkerLoadApplication);

//! Unloads an application, and removes its booting information.
struct WorkerUnloadApplication {
  ApplicationId application_id;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_UNLOAD_APPLICATION,
                 WorkerUnloadApplication);

//! Loads one or many partitions, and prepares them for receiving data.
struct WorkerLoadPartitions {
  ApplicationId application_id;
  PartitionList partition_list;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, partition_list);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_LOAD_PARTITIONS, WorkerLoadPartitions);

//! Unloads one or many partitions, and cleans up their data.
struct WorkerUnloadPartitions {
  ApplicationId application_id;
  PartitionList partition_list;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, partition_list);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_UNLOAD_PARTITIONS,
                 WorkerUnloadPartitions);

//! Gets prepared of migrated in partitions, such that they are ready to receive
// data. Responds when the preparation is ready and when the migrating is done.
struct WorkerMigrateInPartitions {
  ApplicationId application_id;
  PartitionList partition_list;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, partition_list);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_MIGRATE_IN_PARTITIONS,
                 WorkerMigrateInPartitions);

//! Starts migrating out partitions.
struct WorkerMigrateOutPartitions {
  struct PartitionMigrateRecord {
    VariableGroupId variable_group_id;
    PartitionId partition_id;
    WorkerId to_worker_id;
    template <typename Archive>
    void serialize(Archive& archive) {  // NOLINT
      archive(variable_group_id, partition_id, to_worker_id);
    }
  };
  ApplicationId application_id;
  std::list<PartitionMigrateRecord> migrate_out_partitions;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, migrate_out_partitions);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_MIGRATE_OUT_PARTITIONS,
                 WorkerMigrateOutPartitions);

//! Asks a worker to report running status of partitions.
struct WorkerReportStatusOfPartitions {
  ApplicationId application_id;
  PartitionList partition_list;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, partition_list);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_REPORT_STATUS_OF_PARTITIONS,
                 WorkerReportStatusOfPartitions);

//! Asks a worker to report its running status.
struct WorkerReportRunningStatus {
  template <typename Archive>
  void serialize(Archive&) {  // NOLINT
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_REPORT_RUNNING_STATUS,
                 WorkerReportRunningStatus);

//! Changes the priority of an application.
struct WorkerChangeApplicationPriority {
  ApplicationId application_id;
  PriorityLevel priority_level;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, priority_level);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_CHANGE_APPLICATION_PRIORITY,
                 WorkerChangeApplicationPriority);

/*
 * Progress control.
 */
//! Asks a worker to pause the execution of corresponding partitions.
struct WorkerPauseExecution {
  ApplicationId application_id;
  PartitionList partition_list;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, partition_list);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_PAUSE_EXECUTION, WorkerPauseExecution);

//! Asks a worker to installs a barrier on corresponding partitions.
struct WorkerInstallBarrier {
  ApplicationId application_id;
  PartitionList partition_list;
  StageId barrier_stage;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, partition_list, barrier_stage);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_INSTALL_BARRIER, WorkerInstallBarrier);

//! Asks a worker to release the barrier of corresponding partitions.
struct WorkerReleaseBarrier {
  ApplicationId application_id;
  PartitionList partition_list;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, partition_list);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_RELEASE_BARRIER, WorkerReleaseBarrier);

// NOTE: By chinmayee
//! Asks worker to proceed, update partitions done.
struct WorkerUpdatePlacementDone {
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  int partition_id;
  template <typename Archive>
  void serialize(Archive &archive) {  // NOLINT
    archive(application_id, variable_group_id, partition_id);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_UPDATE_PLACEMENT_DONE,
                 WorkerUpdatePlacementDone);

/*
 * Worker to controller commands.
 */

/**
 * The running statistics of a partition.
 */
struct RunningStats {
  //! Earliest unfinished stage may be INVALID.
  StageId earliest_unfinished_stage_id;
  //! Last finished stage may be COMPLETE.
  StageId last_finished_stage_id;
  //! double is total time.
  double migration_time;
  //! double is a timestamp.
  std::map<StatementId, std::pair<double, int>> statement_stats;
  //! double is a timestamp.
  std::map<StageId, std::pair<StatementId, double>> timestamp_stats;
  //! double is cycles since last track statement.
  std::map<StageId, std::pair<StatementId, double>> cycles_track_stats;
  //! double is the cycle spent on the following stages in the same loop.
  std::map<StageId, std::pair<StatementId, double>> cycle_stats;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(earliest_unfinished_stage_id, last_finished_stage_id);
    archive(migration_time);
    archive(statement_stats, timestamp_stats, cycles_track_stats, cycle_stats);
  }
};

//! Responds that migration in is prepared.
struct ControllerRespondMigrationInPrepared {
  WorkerId from_worker_id;
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id);
    archive(application_id, variable_group_id, partition_id);
  }
};
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_RESPOND_MIGRATION_IN_PREPARED,
                 ControllerRespondMigrationInPrepared);

//! Responds that migration in is completed.
struct ControllerRespondMigrationInDone {
  WorkerId from_worker_id;
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id);
    archive(application_id, variable_group_id, partition_id);
  }
};
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_RESPOND_MIGRATION_IN_DONE,
                 ControllerRespondMigrationInDone);

//! Responds that migration out is completed.
struct ControllerRespondMigrationOutDone {
  WorkerId from_worker_id;
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  //! Sends the running stats when a partition is migrated out.
  RunningStats running_stats;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id);
    archive(application_id, variable_group_id, partition_id);
    archive(running_stats);
  }
};
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_RESPOND_MIGRATION_OUT_DONE,
                 ControllerRespondMigrationOutDone);

//! Responds that a partition is complete.
struct ControllerRespondPartitionDone {
  WorkerId from_worker_id;
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  //! Sends the running stats when a partition is complete.
  RunningStats running_stats;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id);
    archive(application_id, variable_group_id, partition_id);
    archive(running_stats);
  }
};
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_RESPOND_PARTITION_DONE,
                 ControllerRespondPartitionDone);

//! Responds with the running status of a partition, only when requested by the
// controller.
struct ControllerRespondStatusOfPartition {
  WorkerId from_worker_id;
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  RunningStats running_stats;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id);
    archive(application_id, variable_group_id, partition_id);
    archive(running_stats);
  }
};
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_RESPOND_STATUS_OF_PARTITION,
                 ControllerRespondStatusOfPartition);
//! Responds with the configuration/running status of a worker.
struct ControllerRespondStatusOfWorker {
  WorkerId from_worker_id;
  int num_cores = -1;
  //! CPU utilization percentage of all applications (might not be Canary).
  double all_cpu_usage_percentage = 0;
  //! CPU utilization percentage of Canary.
  double canary_cpu_usage_percentage = 0;
  //! All available memory space in GB.
  double available_memory_gb = 0;
  //! Memory space used by Canary in GB.
  double used_memory_gb = 0;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id, num_cores);
    archive(all_cpu_usage_percentage, canary_cpu_usage_percentage,
            available_memory_gb, used_memory_gb);
  }
};
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_RESPOND_STATUS_OF_WORKER,
                 ControllerRespondStatusOfWorker);

//! Responds when the execution of a partition is paused.
struct ControllerRespondPauseExecution {
  WorkerId from_worker_id;
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  RunningStats running_stats;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id);
    archive(application_id, variable_group_id, partition_id);
    archive(running_stats);
  }
};
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_RESPOND_PAUSE_EXECUTION,
                 ControllerRespondPauseExecution);

//! Responds when a barrier is reached.
struct ControllerRespondReachBarrier {
  WorkerId from_worker_id;
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  RunningStats running_stats;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id);
    archive(application_id, variable_group_id, partition_id);
    archive(running_stats);
  }
};
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_RESPOND_REACH_BARRIER,
                 ControllerRespondReachBarrier);

// NOTE: By chinmayee
//! Send new set of partitions and times when they should become active.
struct ControllerSendPartitionHistory {
  WorkerId from_worker_id;
  ApplicationId application_id;
  int num_partitions;
  int history_len;
  float last_time;
  std::vector<float> times;
  std::vector< std::vector<int> > history;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id, application_id);
    archive(num_partitions, history_len, last_time);
    archive(times);
    archive(history);
  }
};  // struct ControllerSendPartitionHistory
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_SEND_PARTITION_HISTORY,
                 ControllerSendPartitionHistory);

// NOTE: By chinmayee
//! Send update partition message to controller.
struct ControllerUpdatePlacementForTime {
  WorkerId from_worker_id;
  int num_partitions;
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  int partition_id;
  float time;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id, num_partitions);
    archive(application_id, variable_group_id, partition_id);
    archive(time);
  }
};  // struct ControllerUpdatePlacementForTime
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_UPDATE_PLACEMENT_FOR_TIME,
                 ControllerUpdatePlacementForTime);

}  // namespace message
}  // namespace canary
#endif  // CANARY_SRC_MESSAGE_COMMAND_MESSAGE_H_
