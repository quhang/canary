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
 * @file src/controller/controller_scheduler.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class ControllerScheduler.
 */

#ifndef CANARY_SRC_CONTROLLER_CONTROLLER_SCHEDULER_H_
#define CANARY_SRC_CONTROLLER_CONTROLLER_SCHEDULER_H_

#include <map>
#include <set>
#include <string>

#include "shared/canary_internal.h"

#include "controller/controller_communication_interface.h"
#include "message/message_include.h"
#include "shared/canary_application.h"
#include "shared/network.h"
#include "shared/partition_map.h"

namespace canary {

class ControllerSchedulerBase : public ControllerReceiveCommandInterface {
 public:
  //! Constructor.
  ControllerSchedulerBase() {}

  //! Destructor.
  virtual ~ControllerSchedulerBase() {}

  //! Initialize.
  void Initialize(network::EventMainThread* event_main_thread,
                  ControllerSendCommandInterface* send_command_interface);

  //! Called when receiving a command. The message header is kept, and the
  // buffer ownership is transferred.
  void ReceiveCommand(struct evbuffer* buffer) override;

  //! Called when a worker is down, even if it is shut down by the controller.
  void NotifyWorkerIsDown(WorkerId worker_id) override;

  //! Called when a worker is up. The up notification and down notification are
  // paired.
  void NotifyWorkerIsUp(WorkerId worker_id) override;

 protected:
  //! Called when receiving commands from a worker.
  virtual void InternalReceiveCommand(struct evbuffer* buffer) = 0;

  //! Called when a worker is down, even if it is shut down by the controller.
  virtual void InternalNotifyWorkerIsDown(WorkerId worker_id) = 0;

  //! Called when a worker is up. The up notification and down notification are
  // paired.
  virtual void InternalNotifyWorkerIsUp(WorkerId worker_id) = 0;

 protected:
  network::EventMainThread* event_main_thread_ = nullptr;
  ControllerSendCommandInterface* send_command_interface_ = nullptr;
};

class ControllerScheduler : public ControllerSchedulerBase {
 protected:
  //! Represents an active worker.
  struct WorkerRecord {
    int num_cores = -1;
    std::map<ApplicationId, std::set<FullPartitionId>> owned_partitions;
    std::set<ApplicationId> loaded_applications;
  };
  //! Represents an active application.
  struct ApplicationRecord {
    std::string binary_location;
    std::string application_parameter;
    void* loading_handle = nullptr;
    CanaryApplication* loaded_application = nullptr;
    const CanaryApplication::VariableGroupInfoMap* variable_group_info_map =
        nullptr;
    PerApplicationPartitionMap per_app_partition_map;
  };

 public:
  ControllerScheduler() {}
  virtual ~ControllerScheduler() {}

 protected:
  //! Called when receiving commands from a worker.
  void InternalReceiveCommand(struct evbuffer* buffer) override;

  //! Called when a worker is down, even if it is shut down by the controller.
  void InternalNotifyWorkerIsDown(WorkerId worker_id) override;

  //! Called when a worker is up. The up notification and down notification are
  // paired.
  void InternalNotifyWorkerIsUp(WorkerId worker_id) override;

 private:
  /*
   * Processes messages from the launcher or workers.
   */
  void ProcessLaunchApplication(message::LaunchApplication* launch_message);
  void ProcessMigrationInPrepared(
      message::ControllerRespondMigrationInPrepared* respond_message);
  void ProcessMigrationInDone(
      message::ControllerRespondMigrationInDone* respond_message);
  void ProcessStatusOfPartition(
      message::ControllerRespondStatusOfPartition* respond_message);
  void ProcessStatusOfWorker(
      message::ControllerRespondStatusOfWorker* respond_message);

  /*
   * Application launching related.
   */
  //! Fills in initial info in the application record.
  void FillInApplicationLaunchInfo(
      const message::LaunchApplication& launch_message,
      ApplicationRecord* application_record);
  //! Assigns partitions to workers.
  void AssignPartitionToWorker(ApplicationRecord* application_record);
  //! Gets the next assigned worker id.
  WorkerId NextAssignWorkerId();
  //! Loads an application on all workers.
  void RequestLoadApplicationOnAllWorkers(
      ApplicationId application_id,
      const ApplicationRecord& application_record);
  //! Updates the partitions each workers own.
  void UpdateWorkerOwnedPartitions(
      ApplicationId application_id,
      const PerApplicationPartitionMap& per_app_partition_map);
  //! Requests workers to load partitions.
  void RequestLoadPartitions(ApplicationId application_id);

 private:
  WorkerId last_assigned_worker_id_ = WorkerId::INVALID;
  ApplicationId next_application_id_ = ApplicationId::FIRST;

 protected:
  std::map<WorkerId, WorkerRecord> worker_map_;
  std::map<ApplicationId, ApplicationRecord> application_map_;
};

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_CONTROLLER_SCHEDULER_H_
