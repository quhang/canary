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
#include <string>
#include <utility>

#include "shared/canary_internal.h"

#include "message/message.h"

namespace canary {
namespace message {

struct TestWorkerCommand {
  std::string test_string;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(test_string);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, TEST_WORKER_COMMAND, TestWorkerCommand);

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

//! Loads an application, and saves its booting information at the worker
// scheduler.
struct WorkerLoadApplication {
  ApplicationId application_id;
  std::string binary_location;
  std::string application_parameter;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, binary_location, application_parameter);
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
  std::list<std::pair<VariableGroupId, PartitionId>> load_partitions;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id);
    archive(load_partitions);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_LOAD_PARTITIONS, WorkerLoadPartitions);

//! Unloads one or many partitions, and cleans up their data.
struct WorkerUnloadPartitions {
  ApplicationId application_id;
  std::list<std::pair<VariableGroupId, PartitionId>> unload_partitions;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, unload_partitions);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_UNLOAD_PARTITIONS,
                 WorkerUnloadPartitions);

//! Gets prepared of migrated in partitions, such that they are ready to receive
// data. Responds when the preparation is ready and when the migrating is done.
struct WorkerMigrateInPartitions {
  ApplicationId application_id;
  std::list<std::pair<VariableGroupId, PartitionId>> migrate_in_partitions;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, migrate_in_partitions);
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
  std::list<std::pair<VariableGroupId, PartitionId>> report_partitions;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, report_partitions);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_REPORT_STATUS_OF_PARTITIONS,
                 WorkerReportStatusOfPartitions);

//! Asks a worker to report running status of the worker.
struct WorkerReportStatusOfWorker {
  template <typename Archive>
  void serialize(Archive&) {  // NOLINT
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_REPORT_STATUS_OF_WORKER,
                 WorkerReportStatusOfWorker);

//! Controls the behavior of partitions.
struct WorkerControlPartitions {
  std::list<std::pair<VariableGroupId, PartitionId>> control_partitions;
  StageId next_barrier_stage;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(control_partitions, next_barrier_stage);
  }
};
REGISTER_MESSAGE(WORKER_COMMAND, WORKER_CONTROL_PARTITIONS,
                 WorkerControlPartitions);

/*
 * Worker to controller commands.
 */

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

//! Responds with the running status of a partition.
struct ControllerRespondStatusOfPartition {
  WorkerId from_worker_id;
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  // Cycles.
  // Barrier reached.
  // Critical breakpoint.
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id);
    archive(application_id, variable_group_id, partition_id);
  }
};
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_RESPOND_STATUS_OF_PARTITION,
                 ControllerRespondStatusOfPartition);

//! Responds with the running status of a worker.
struct ControllerRespondStatusOfWorker {
  WorkerId from_worker_id;
  int num_cores = -1;
  // CPU utilization.
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id, num_cores);
  }
};
REGISTER_MESSAGE(CONTROLLER_COMMAND, CONTROLLER_RESPOND_STATUS_OF_WORKER,
                 ControllerRespondStatusOfWorker);

}  // namespace message
}  // namespace canary
#endif  // CANARY_SRC_MESSAGE_COMMAND_MESSAGE_H_
