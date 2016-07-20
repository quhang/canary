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
 * @file src/worker/worker_scheduler.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class WorkerScheduler.
 */

#ifndef CANARY_SRC_WORKER_WORKER_SCHEDULER_H_
#define CANARY_SRC_WORKER_WORKER_SCHEDULER_H_

#include <list>
#include <map>
#include <string>
#include <vector>

#include "shared/canary_internal.h"

#include "message/message_include.h"
#include "shared/canary_application.h"
#include "shared/network.h"
#include "shared/partition_map.h"
#include "worker/worker_communication_interface.h"
#include "worker/worker_light_thread_context.h"

namespace canary {

/**
 * A worker scheduler schedules local execution and runs in the same thread as
 * the worker communication manager for reducing sychronization overhead.
 */
class WorkerSchedulerBase : public WorkerReceiveCommandInterface,
                            public WorkerReceiveDataInterface {
 protected:
  //! The record of an application.
  struct ApplicationRecord {
    ApplicationId application_id = ApplicationId::INVALID;
    std::string binary_location;
    std::string application_parameter;
    StageId first_barrier_stage = StageId::INVALID;
    int local_partitions = 0;
    void* loading_handle = nullptr;
    CanaryApplication* loaded_application = nullptr;
  };

 public:
  //! Constructor.
  WorkerSchedulerBase();
  //! Destroctor.
  virtual ~WorkerSchedulerBase();
  //! Initializes the worker scheduler.
  void Initialize(WorkerSendCommandInterface* send_command_interface,
                  WorkerSendDataInterface* send_data_interface);
  /*
   * Callback exposed to the worker communication manager and the worker data
   * router. These methods are called in the synchronous context.
   */
  //! Delivers the routed data to a thread context, and returns false if no
  // thread context is found.
  bool ReceiveRoutedData(ApplicationId application_id,
                         VariableGroupId variable_group_id,
                         PartitionId partition_id, StageId stage_id,
                         struct evbuffer* buffer) override;
  //! Delivers direct data.
  void ReceiveDirectData(struct evbuffer* buffer) override;
  //! Processes worker commands.
  void ReceiveCommandFromController(struct evbuffer* buffer) override;
  //! Assigns a worker id, and starts execution.
  void AssignWorkerId(WorkerId worker_id) override;

 private:
  /*
   * Processing various commands from the controller. These methods are
   * called in the synchronous context.
   */
  //! Loads an application.
  void ProcessLoadApplication(
      const message::WorkerLoadApplication& worker_command);
  //! Unloads an application.
  void ProcessUnloadApplication(
      const message::WorkerUnloadApplication& worker_command);
  //! Loads partitions.
  void ProcessLoadPartitions(
      const message::WorkerLoadPartitions& worker_command);
  //! Unloads partitions.
  void ProcessUnloadPartitions(
      const message::WorkerUnloadPartitions& worker_command);
  //! TODO(quhang): not implemented.
  void ProcessMigrateInPartitions(
      const message::WorkerMigrateInPartitions& worker_command);
  //! TODO(quhang): not implemented.
  void ProcessMigrateOutPartitions(
      const message::WorkerMigrateOutPartitions& worker_command);
  //! Asks a partition to report status.
  void ProcessReportStatusOfPartitions(
      const message::WorkerReportStatusOfPartitions& worker_command);
  //! Asks a partition to release a barrier.
  void ProcessReleaseBarrier(
      const message::WorkerReleaseBarrier& worker_command);

 protected:
  /*
   * Methods implemented by subclass to change the behavior of the scheduler.
   */
  //! Starts execution threads.
  virtual void StartExecution() = 0;
  //! Loads the application binary.
  virtual void LoadApplicationBinary(ApplicationRecord* application_record) = 0;
  //! Unloads the application binary.
  virtual void UnloadApplicationBinary(
      ApplicationRecord* application_record) = 0;
  //! Loads a partition and returns its thread context.
  virtual WorkerLightThreadContext* LoadPartition(
      FullPartitionId full_partition_id) = 0;
  //! Unloads a partition and wraps up its thread context.
  virtual void UnloadPartition(WorkerLightThreadContext* thread_context) = 0;
  /*
   * Execution thread control. Called in asynchronous context.
   */
  //! Activates a thread context.
  void ActivateThreadContext(WorkerLightThreadContext* thread_context);
  //! Retrieves an activated thread context.
  WorkerLightThreadContext* GetActivatedThreadContext();
  //! Execution routine.
  static void* ExecutionRoutine(void* arg);

 protected:
  /*
   * Scheduling-related data structure.
   */
  //! Thread scheduling sychronization lock and condition variable.
  pthread_mutex_t scheduling_lock_;
  pthread_cond_t scheduling_cond_;
  //! All thread contexts.
  std::map<FullPartitionId, WorkerLightThreadContext*> thread_map_;
  //! Activated thread contexts.
  std::list<WorkerLightThreadContext*> activated_thread_queue_;
  //! Execution thread handlers.
  std::vector<pthread_t> thread_handle_list_;
  /*
   * Other data structure.
   */
  std::map<ApplicationId, ApplicationRecord> application_record_map_;
  WorkerSendCommandInterface* send_command_interface_ = nullptr;
  WorkerSendDataInterface* send_data_interface_ = nullptr;
  //! Whether a worker id is assigned.
  bool is_ready_ = false;
  WorkerId self_worker_id_ = WorkerId::INVALID;
  int num_cores_ = -1;
};

class WorkerScheduler : public WorkerSchedulerBase {
 public:
  WorkerScheduler() {}
  virtual ~WorkerScheduler() {}

 protected:
  //! Starts execution threads.
  void StartExecution() override;

  //! Loads the application binary.
  void LoadApplicationBinary(ApplicationRecord* application_record) override;

  //! Unloads the application binary.
  void UnloadApplicationBinary(ApplicationRecord* application_record) override;

  //! Loads a partition and returns its thread context.
  WorkerLightThreadContext* LoadPartition(
      FullPartitionId full_partition_id) override;

  //! Unloads a partition and wraps up its thread context.
  void UnloadPartition(WorkerLightThreadContext* thread_context) override;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_SCHEDULER_H_
