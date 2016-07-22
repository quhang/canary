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
    // Caution: after saving, the buffer is destroyed.
    template <typename Archive>
    void save(Archive& archive) const {  // NOLINT
      archive(buffer_list.size());
      for (auto buffer : buffer_list) {
        archive(RawEvbuffer{buffer});
      }
      archive(expected_buffer);
    }
    template <typename Archive>
    void load(Archive& archive) {  // NOLINT
      size_t buffer_size;
      archive(buffer_size);
      for (size_t i = 0; i < buffer_size; ++i) {
        RawEvbuffer raw_buffer;
        archive(raw_buffer);
        buffer_list.push_back(raw_buffer.buffer);
      }
      archive(expected_buffer);
    }
  };
  //! The buffer storing a command.
  struct CommandBuffer {
    StageId stage_id;
    struct evbuffer* command;
    // Caution: after saving, the buffer is destroyed.
    template <typename Archive>
    void save(Archive& archive) const {  // NOLINT
      archive(stage_id);
      archive(RawEvbuffer{command});
    }
    template <typename Archive>
    void load(Archive& archive) {  // NOLINT
      archive(stage_id);
      RawEvbuffer raw_buffer;
      archive(raw_buffer);
      command = raw_buffer.buffer;
    }
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

 public:
  virtual void load(CanaryInputArchive& archive) {  // NOLINT
    archive(command_list_);
    archive(stage_buffer_map_);
    archive(ready_stages_);
  }
  virtual void save(CanaryOutputArchive& archive) const {  // NOLINT
    // States are destroyed after serialization.
    archive(command_list_);
    archive(stage_buffer_map_);
    archive(ready_stages_);
  }
  // Used after SAVE to clear all memory.
  virtual void clear_memory() {
    if (memory_cleared_) {
      return;
    }
    memory_cleared_ = true;
    command_list_.clear();
    stage_buffer_map_.clear();
    ready_stages_.clear();
  }

 private:
  friend class WorkerSchedulerBase;
  // Caution: TRANSIENT.
  WorkerId worker_id_ = WorkerId::INVALID;
  // Caution: TRANSIENT.
  ApplicationId application_id_ = ApplicationId::INVALID;
  // Caution: TRANSIENT.
  VariableGroupId variable_group_id_ = VariableGroupId::INVALID;
  // Caution: TRANSIENT.
  PartitionId partition_id_ = PartitionId::INVALID;
  // Caution: TRANSIENT.
  std::function<void()> activate_callback_;
  // Caution: TRANSIENT.
  WorkerSendCommandInterface* send_command_interface_ = nullptr;
  // Caution: TRANSIENT.
  WorkerSendDataInterface* send_data_interface_ = nullptr;
  // Caution: TRANSIENT.
  const CanaryApplication* canary_application_ = nullptr;
  //! Synchronization lock.
  // Caution: TRANSIENT.
  pthread_mutex_t internal_lock_;
  // Caution: TRANSIENT.
  bool running_ = false;
  // Caution: TRANSIENT.
  bool memory_cleared_ = false;

  /*
   * States that need serialization.
   */
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
  //! Builds running stats.
  void BuildStats(message::RunningStats* running_stats);
  //! Fills in running stats into a command.
  template <typename T>
  void FillInStats(T* report);
  //! Processes an initialization command.
  void ProcessInitCommand(struct evbuffer* command);
  //! Processes a control flow decision.
  void ProcessControlFlowDecision(struct evbuffer* command);
  //! Processes a command that requests running stats.
  void ProcessRequestReport();
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
  //! Deserializes a control flow decision.
  void DeserializeControlFlowDecision(struct evbuffer* buffer,
                                      StageId* stage_id, bool* decision);
  //! Allocates data partitions.
  void AllocatePartitionData();
  //! Deallocates data partitions.
  void DeallocatePartitionData();

 public:
  void load(CanaryInputArchive& archive) override {  // NOLINT
    WorkerLightThreadContext::load(archive);
    archive(stage_graph_);
    archive(pending_gather_stages_);
    archive(is_in_barrier_);
    AllocatePartitionData();
    for (auto& pair : local_partition_data_) {
      VariableId variable_id;
      archive(variable_id);
      CHECK(pair.first == variable_id);
      pair.second->Deserialize(archive);
    }
  }
  void save(CanaryOutputArchive& archive) const override {  // NOLINT
    WorkerLightThreadContext::save(archive);
    archive(stage_graph_);
    archive(pending_gather_stages_);
    archive(is_in_barrier_);
    for (auto& pair : local_partition_data_) {
      archive(pair.first);
      pair.second->Serialize(archive);
      pair.second->Finalize();
    }
  }
  void clear_memory() override {
    WorkerLightThreadContext::clear_memory();
    pending_gather_stages_.clear();
    local_partition_data_.clear();
    stage_graph_.~StageGraph();
  }

 private:
  StageGraph stage_graph_;
  std::map<StageId, StatementId> pending_gather_stages_;
  std::map<VariableId, std::unique_ptr<PartitionData>> local_partition_data_;
  bool is_in_barrier_ = false;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_LIGHT_THREAD_CONTEXT_H_
