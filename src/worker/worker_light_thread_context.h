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
 * @file src/worker/worker_light_thread_context.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class WorkerLightThreadContext.
 */

#ifndef CANARY_SRC_WORKER_WORKER_LIGHT_THREAD_CONTEXT_H_
#define CANARY_SRC_WORKER_WORKER_LIGHT_THREAD_CONTEXT_H_

#include <list>
#include <map>

#include "shared/canary_internal.h"

#include "shared/canary_application.h"
#include "worker/canary_task_context.h"
#include "worker/stage_graph.h"
#include "worker/worker_communication_interface.h"

namespace canary {

namespace message {
// Forward declaration.
struct RunningStats;
}  // namespace message

/**
 * These messages are for internal command communication.
 */
namespace internal_message {}  // namespace internal_message

/**
 * The execution context of a lightweight thread, which is responsible for
 * execution of a partition. This base class stores basic metadata, and includes
 * logic to handle command and data delivery.
 */
class WorkerLightThreadContext {
 private:
  //! The buffer storing received data of a stage.
  struct StageBuffer {
    std::list<struct evbuffer*> buffer_list;
    int expected_buffer = -1;
  };
  //! The buffer storing a command.
  struct CommandBuffer {
    StageId stage_id;
    struct evbuffer* command;
  };

 public:
  //! Constructor.
  WorkerLightThreadContext();
  //! Destructor.
  virtual ~WorkerLightThreadContext();
  //! Initializes the light thread.
  virtual void Initialize() = 0;
  //! Finalizes the light thread.
  virtual void Finalize() = 0;
  //! Runs the thread.
  virtual void Run() = 0;

 public:
  //! Gets/sets the metadata of the thread.
  ApplicationId get_application_id() const { return application_id_; }
  void set_application_id(ApplicationId application_id) {
    application_id_ = application_id;
  }
  VariableGroupId get_variable_group_id() const { return variable_group_id_; }
  void set_variable_group_id(VariableGroupId variable_group_id) {
    variable_group_id_ = variable_group_id;
  }
  PartitionId get_partition_id() const { return partition_id_; }
  void set_partition_id(PartitionId partition_id) {
    partition_id_ = partition_id;
  }
  PriorityLevel get_priority_level() const { return priority_level_; }
  void set_priority_level(PriorityLevel priority_level) {
    priority_level_ = priority_level;
  }
  WorkerId get_worker_id() const { return worker_id_; }
  void set_worker_id(WorkerId worker_id) { worker_id_ = worker_id; }
  void set_activate_callback(std::function<void()> activate_callback) {
    activate_callback_ = std::move(activate_callback);
  }
  WorkerSendCommandInterface* get_send_command_interface() {
    return send_command_interface_;
  }
  void set_send_command_interface(WorkerSendCommandInterface* interface) {
    send_command_interface_ = interface;
  }
  WorkerSendDataInterface* get_send_data_interface() {
    return send_data_interface_;
  }
  void set_send_data_interface(WorkerSendDataInterface* interface) {
    send_data_interface_ = interface;
  }
  const CanaryApplication* get_canary_application() const {
    return canary_application_;
  }
  void set_canary_application(const CanaryApplication* canary_application) {
    canary_application_ = canary_application;
  }

  //! Tries to enter the execution context of the thread. Returns true if there
  // is data to process.
  bool Enter();
  //! Tries to exit the execution context of the thread. Returns true if there
  // is no data to process.
  bool Exit();
  //! Forces the execution context to exit.
  void ForceExit();
  //! Delivers a message.
  void DeliverMessage(StageId stage_id, struct evbuffer* buffer);

 protected:
  //! Registers how many messages are expected for a message.
  void RegisterReceivingData(StageId stage_id, int num_message);
  //! Retrieves a command.
  bool RetrieveCommand(StageId* stage_id, struct evbuffer** command);
  //! Retrieves the buffer of a stage.
  bool RetrieveData(StageId* stage_id,
                    std::list<struct evbuffer*>* buffer_list);

 private:
  //! Metadata.
  WorkerId worker_id_ = WorkerId::INVALID;
  ApplicationId application_id_ = ApplicationId::INVALID;
  VariableGroupId variable_group_id_ = VariableGroupId::INVALID;
  PartitionId partition_id_ = PartitionId::INVALID;
  PriorityLevel priority_level_ = PriorityLevel::INVALID;
  std::function<void()> activate_callback_;
  WorkerSendCommandInterface* send_command_interface_ = nullptr;
  WorkerSendDataInterface* send_data_interface_ = nullptr;
  const CanaryApplication* canary_application_ = nullptr;
  //! Synchronization lock.
  pthread_mutex_t internal_lock_;
  bool running_ = false;
  //! Received commands.
  std::list<CommandBuffer> command_list_;
  //! Received data.
  std::map<StageId, StageBuffer> stage_buffer_map_;
  //! Ready stages.
  std::list<StageId> ready_stages_;
};

class WorkerExecutionContext : public WorkerLightThreadContext {
 public:
  WorkerExecutionContext() {}
  virtual ~WorkerExecutionContext() {}

  //! Initializes the light thread.
  void Initialize() override;
  //! Finalizes the light thread.
  void Finalize() override;
  //! Runs the thread.
  void Run() override;

 private:
  //! Builds running status.
  void BuildStats(message::RunningStats* running_stats);
  //! Processes an initialization command.
  void ProcessInitCommand(struct evbuffer* command);
  //! Processes a control flow decision.
  void ProcessControlFlowDecision(struct evbuffer* command);
  //! Processes a command that requests running stats.
  void ProcessRequestReport();

  //! Runs the second step of a gather task.
  void RunGatherStage(StageId stage_id, StatementId statement_id,
                      std::list<struct evbuffer*>* buffer_list);
  //! Runs a stage, including the first step of a gather task.
  void RunStage(StageId stage_id, StatementId statement_id);
  //! Prepares a task context
  void PrepareTaskContext(
      StageId stage_id, StatementId statement_id,
      const CanaryApplication::StatementInfo& statement_info,
      CanaryTaskContext* task_context);
  //! Serializes a control flow decision.
  struct evbuffer* SerializeControlFlowDecision(StageId stage_id,
                                                bool decision);
  //! Deserializes a control flow decision.
  void DeserializeControlFlowDecision(struct evbuffer* buffer,
                                      StageId* stage_id, bool* decision);
  //! Allocates data partitions.
  void AllocatePartitionData();
  //! Deallocates data partitions.
  void DeallocatePartitionData();

 private:
  StageGraph stage_graph_;
  std::map<StageId, StatementId> pending_gather_stages_;
  std::map<VariableId, PartitionData*> local_partition_data_;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_LIGHT_THREAD_CONTEXT_H_
