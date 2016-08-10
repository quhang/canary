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
#include <memory>

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

namespace internal_message {

/**
 * These messages are for internal command communication.
 */
struct InitCommand {
  StageId first_barrier_stage;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(first_barrier_stage);
  }
};
struct ControlDecisionCommand {
  StageId stage_id;
  bool decision;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(stage_id, decision);
  }
};
struct MigrateOutCommand {
  WorkerId to_worker_id;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(to_worker_id);
  }
};

template <typename CommandType>
struct evbuffer* to_buffer(const CommandType& command) {
  struct evbuffer* result = evbuffer_new();
  CanaryOutputArchive archive(result);
  archive(command);
  return result;
}

template <typename CommandType>
CommandType to_command(struct evbuffer* buffer) {
  CommandType result;
  {
    CanaryInputArchive archive(buffer);
    archive(result);
  }
  evbuffer_free(buffer);
  return std::move(result);
}
}  // namespace internal_message

class WorkerSchedulerBase;

/**
 * The execution context of a lightweight thread, which is responsible for
 * execution of a partition. This base class stores basic metadata, and includes
 * logic to handle command and data delivery.
 */
class WorkerLightThreadContext {
  friend class WorkerSchedulerBase;

 private:
  //! The buffer storing received data of a stage.
  struct StageBuffer {
    std::list<struct evbuffer*> buffer_list;
    int expected_buffer = -1;
  };
  //! The buffer storing an internal command.
  struct CommandBuffer {
    StageId stage_id;
    struct evbuffer* command;
  };

 public:
  //! Constructor.
  WorkerLightThreadContext();
  //! Destructor.
  virtual ~WorkerLightThreadContext();
  //! Initializes the thread context.
  virtual void Initialize() = 0;
  //! Finalizes the thread context.
  virtual void Finalize() = 0;
  //! Runs the thread.
  virtual void Run() = 0;
  //! Reports its running stat.
  virtual void Report() = 0;

 public:
  //! Gets the metadata of the thread.
  ApplicationId get_application_id() const { return application_id_; }
  VariableGroupId get_variable_group_id() const { return variable_group_id_; }
  PartitionId get_partition_id() const { return partition_id_; }
  WorkerId get_worker_id() const { return worker_id_; }
  WorkerSendCommandInterface* get_send_command_interface() {
    return send_command_interface_;
  }
  WorkerSendDataInterface* get_send_data_interface() {
    return send_data_interface_;
  }
  const CanaryApplication* get_canary_application() const {
    return canary_application_;
  }
  WorkerSchedulerBase* get_worker_scheduler() { return worker_scheduler_; }
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

 public:
  virtual void load(CanaryInputArchive& archive);         // NOLINT
  virtual void save(CanaryOutputArchive& archive) const;  // NOLINT

 private:
  // Used for scheduling, protected by the lock of the worker scheduler.
  bool is_killed_ = false;
  bool is_running_ = false;
  bool need_report_ = false;
  bool need_process_ = false;
  bool is_in_priority_queue_ = false;
  WorkerId worker_id_ = WorkerId::INVALID;
  ApplicationId application_id_ = ApplicationId::INVALID;
  VariableGroupId variable_group_id_ = VariableGroupId::INVALID;
  PartitionId partition_id_ = PartitionId::INVALID;
  WorkerSendCommandInterface* send_command_interface_ = nullptr;
  WorkerSendDataInterface* send_data_interface_ = nullptr;
  const CanaryApplication* canary_application_ = nullptr;
  WorkerSchedulerBase* worker_scheduler_ = nullptr;
  //! Synchronization lock.
  mutable pthread_mutex_t internal_lock_;

  /*
   * States that need serialization.
   */
  //! Received commands.
  mutable std::list<CommandBuffer> command_list_;
  //! Received data.
  mutable std::map<StageId, StageBuffer> stage_buffer_map_;
  //! Ready stages.
  mutable std::list<StageId> ready_stages_;
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
  //! Reports its running stat.
  void Report() override;

 private:
  //! Runs all commands.
  void RunCommands();
  //! Runs one stage, and returns whether there might be more stages to run.
  bool RunOneStage();
  //! Builds running stats.
  void BuildStats(message::RunningStats* running_stats);
  //! Fills in running stats into a command.
  template <typename T>
  void FillInStats(T* report);
  //! Processes an initialization command.
  void ProcessInitCommand(struct evbuffer* command);
  //! Processes a control flow decision.
  void ProcessControlFlowDecision(struct evbuffer* command);
  //! Processes a migrate in command.
  void ProcessMigrateIn(struct evbuffer* command);
  //! Processes a migrate out command.
  void ProcessMigrateOut(struct evbuffer* command);
  //! Processes a command that releases a buffer.
  void ProcessReleaseBarrier();

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
  //! Allocates data partitions.
  void AllocatePartitionData();
  //! Deallocates data partitions.
  void DeallocatePartitionData();

 public:
  //! Deserialization function.
  void load(CanaryInputArchive& archive) override;  // NOLINT
  //! Serialization function. Data are destroyed after serialization.
  void save(CanaryOutputArchive& archive) const override;  // NOLINT

 private:
  StageGraph stage_graph_;
  std::map<StageId, StatementId> pending_gather_stages_;
  std::map<VariableId, std::unique_ptr<PartitionData>> local_partition_data_;
  enum PartitionState : int32_t {
    UNINITIALIZED = 0,
    RUNNING,
    PAUSED,
    IN_BARRIER,
    MIGRATED,
    COMPLETE
  };
  PartitionState partition_state_ = PartitionState::UNINITIALIZED;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_LIGHT_THREAD_CONTEXT_H_
